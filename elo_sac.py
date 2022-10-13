import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
from encoder import make_encoder
from decoder import make_decoder

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()


        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)
            
            
class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)
            
class EloCombo(nn.Module):
    def __init__(self,
                 obs_shape,
                 action_shape,
                 encoder_feature_dim,
                 hidden_dim,
                 critic,
                 critic_target,
                 weights,
                 device):
        super(EloCombo, self).__init__()
        
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder 
        self.loss_weights = weights
        self.device = device
        self.decoder_latent_lambda = 1e-6
        decoder_type = 'pixel'
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        print(f"encoder_feature_dim: {encoder_feature_dim}, hidden_dim: {hidden_dim}")
        # CURL
        self.W = nn.Parameter(torch.rand(encoder_feature_dim, encoder_feature_dim))
        
        mid_hidden_dim = hidden_dim // 4
        fin_hidden_dim = hidden_dim // 8
        
        # DINO
        self.proj = MLP(encoder_feature_dim, projection_size=fin_hidden_dim,  hidden_size=mid_hidden_dim)
        self.proj_momentum = MLP(encoder_feature_dim, projection_size=fin_hidden_dim,  hidden_size=mid_hidden_dim)
        
        self.proj.apply(weight_init)
        self.proj_momentum.load_state_dict(self.proj.state_dict())
        
        self.register_buffer("center", torch.zeros(1, fin_hidden_dim))
        self.teacher_temp = 0.04
        self.student_temp = 0.1
        self.center_momentum = 0.9

        
        
        # AE
        self.decoder = make_decoder(
            decoder_type, obs_shape, encoder_feature_dim, num_layers=4,
            num_filters=32
        ).to(device)
        
        self.decoder.apply(weight_init)
        
        # Predict reward
        self.pred_a = nn.Sequential(
            nn.Linear(action_shape[0], mid_hidden_dim), nn.ReLU()
        )
        
        self.pred_fr = nn.Sequential(
            nn.Linear(encoder_feature_dim + mid_hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, encoder_feature_dim + 1)
        )
        self.pred_a.apply(weight_init)
        self.pred_fr.apply(weight_init)
        
        # Extract action reward
        self.extract_ar = nn.Sequential(
            nn.Linear(encoder_feature_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_shape[0] + 1)
        )
        self.extract_ar.apply(weight_init)
        
        # Rotation CLS
        self.rot_cls = nn.Sequential(
            nn.Linear(encoder_feature_dim, encoder_feature_dim),
            nn.ReLU(),
            nn.Linear(encoder_feature_dim, 4), # 4 fold rotation prediction
        )
        self.rot_cls.apply(weight_init)
        

    def compute_loss(self, obs1, obs2, action, reward, next_obs, L, step):
        z1, z2 = self.encoder(obs1), self.encoder(obs2)
        # z_next = self.encoder(next_obs)
        
        with torch.no_grad():
            z1_tgt, z2_tgt = self.encoder_target(obs1), self.encoder_target(obs2)
            z_next = self.encoder(next_obs)
            
        def get_curl_loss():
            oln = z1
            tgt = z2_tgt
            Wz = torch.matmul(self.W, tgt.T)  # (encoder_feature_dim,B)
            logits = torch.matmul(oln, Wz)  # (B,B)
            logits = logits - torch.max(logits, 1)[0][:, None]
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            loss = self.cross_entropy_loss(logits, labels)
            L.log('train_curl/loss', loss, step)
            return loss
        
        def get_dino_loss():
            student_output = self.proj(torch.cat([z1, z2], dim=0))
            student_out = F.log_softmax(student_output / self.student_temp, dim=1)

            with torch.no_grad():
                teacher_output = self.proj_momentum(torch.cat([z2_tgt, z1_tgt], dim=0))
                teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=1)
            
            loss = torch.sum(-teacher_out * student_out, dim=1).mean()
            L.log('train_dino/loss', loss, step)
            
            # update centering
            self.center = self.center * self.center_momentum + torch.mean(teacher_output, dim=0) * (1 - self.center_momentum)
            return loss
            
        def get_ae_loss():
            target_obs = torch.cat([obs1, obs2], dim=0)
            h = torch.cat([z1, z2], dim=0)
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = utils.preprocess_obs(target_obs)
            
            rec_obs = self.decoder(h)
            rec_loss = F.mse_loss(target_obs, rec_obs)

            # add L2 penalty on latent representation
            # see https://arxiv.org/pdf/1903.12436.pdf
            latent_loss = (0.5 * h.pow(2).sum(1)).mean()
            
            loss = rec_loss + self.decoder_latent_lambda * latent_loss
            L.log('train_ae/loss', loss, step)
            return loss
            
        def get_predict_fr_loss():
            action_h = self.pred_a(action)
            logits = self.pred_fr(torch.cat([z1, action_h], dim=1))
            loss_h = F.mse_loss(logits[:, :-1], z_next)
            loss_r = F.mse_loss(logits[:, -1:], reward)
            loss = (loss_h + loss_r) / 2
            L.log('train_predictfr/loss', loss, step)
            return loss
        
        def get_extract_ar_loss():
            logits = self.extract_ar(torch.cat([z1, z_next], dim=1))
            loss_action = F.mse_loss(logits[:, :-1], action)
            loss_reward = F.mse_loss(logits[:, -1:], reward)
            loss = (loss_action + loss_reward) / 2
            L.log('train_extractar/loss', loss, step)
            return loss
        
        def get_rot_cls_loss():
            b = obs1.size(0)
            labels = torch.arange(4, dtype=torch.long, device=self.device).repeat_interleave(b)
            obs_cat = obs1.repeat(3, 1, 1, 1)
            for i in range(3):
                obs_cat[i * b: (i + 1) * b] = torch.rot90(obs_cat[i * b:(i + 1) * b], i + 1, [-2, -1])
            z1_rot = self.encoder(obs_cat)
            logits = self.rot_cls(torch.cat([z1, z1_rot], dim=0))
            # print(logits.shape, labels.shape)
            loss = F.cross_entropy(logits, labels)
            L.log('train_rotcls/loss', loss, step)
            return loss
        
        
        
        total_loss = 0
        for i, func in enumerate([
            get_curl_loss, 
            get_dino_loss, 
            get_ae_loss, 
            get_predict_fr_loss, 
            get_extract_ar_loss,
            get_rot_cls_loss]):
            if self.loss_weights[i] > 0:
                loss = self.loss_weights[i] * func()
                total_loss += loss
                L.log(f'train_total/loss_{i}', loss, step)
        
        return total_loss
    

class EloSacAgent(object):
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim,
        discount,
        init_temperature,
        alpha_lr,
        alpha_beta,
        actor_lr,
        actor_beta,
        actor_log_std_min,
        actor_log_std_max,
        actor_update_freq,
        critic_lr,
        critic_beta,
        critic_tau,
        critic_target_update_freq,
        encoder_type,
        encoder_feature_dim,
        encoder_lr,
        encoder_tau,
        num_layers,
        num_filters,
        cpc_update_freq,
        log_interval,
        weights
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.encoder_type = encoder_type
        self.loss_weights = weights

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        
        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        
        self.combo = EloCombo(
            obs_shape, 
            action_shape, 
            encoder_feature_dim,
            hidden_dim, 
            self.critic,
            self.critic_target, 
            self.loss_weights, 
            device).to(device)
        
        self.combo_optimizer = torch.optim.Adam(
            self.combo.parameters(), lr=encoder_lr
        )
        
        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.combo.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)
            
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)
 
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=False)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:                                    
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
    def update_combo(self, obs_rl, obs, action, reward, next_obs, L, step):
        self.combo_optimizer.zero_grad()
        loss = self.combo.compute_loss(
            obs_rl, obs, action, reward, next_obs, L, step
        )
        if isinstance(loss, int):
            return 
        loss.backward()
        self.combo_optimizer.step()

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done, aug_obs = replay_buffer.sample_cpc()
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )
            utils.soft_update_params(
                self.combo.proj, self.combo.proj_momentum,
                self.encoder_tau
            )
            
        if step % self.cpc_update_freq == 0 and self.encoder_type == 'pixel':
            self.update_combo(obs, aug_obs, action, reward, next_obs, L, step)
            

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.combo.state_dict(), '%s/elo_%s.pt' % (model_dir, step)
        )


    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.combo.load_state_dict(
            torch.load('%s/elo_%s.pt' % (model_dir, step))
        )
 
