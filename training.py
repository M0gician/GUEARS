import torch
import torch.nn as nn

class Trainer:
    def __init__(self, data, model, optimizer, args):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2)
        self.args = args

        self.user_embedding = nn.Embedding(args['user_count'], args['state_dim'])
        self.item_embedding = nn.Embedding(args['item_count'], args['state_dim'])

    def train_epoch(self):
        loss_total = 0
        for i, train_batch in enumerate(self.data):
            # unpack data
            user_ids, item_ids, rewards, rtg, timesteps = train_batch
            user_ids = user_ids.unsqueeze(-1).repeat(1, item_ids.shape[1])
            rewards = rewards.unsqueeze(-1)
            rtg = rtg.unsqueeze(-1)
            states = self.user_embedding(user_ids)
            actions = self.item_embedding(item_ids)

            timesteps = torch.arange(0, 737).unsqueeze(0).repeat(item_ids.shape[0], 1)
            # set prediction target
            action_target = torch.clone(actions)
            attention_mask = torch.ones_like(rewards).squeeze()

            states = states[:, :320]
            actions = actions[:, :320]
            rewards = rewards[:, :320]
            rtg = rtg[:, :321]
            timesteps = timesteps[:, :320]
            attention_mask = attention_mask[:, :320]
            action_target = actions.clone()
            print(states.shape, actions.shape, rewards.shape, rtg[:,:-1].shape, timesteps.shape, attention_mask.shape)

            # do forward
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
            )

            # compute loss
            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

            loss = self.loss_fn(
                None, action_preds, None,
                None, action_target, None,
            )

            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            self.optimizer.step()
            loss_total += loss.detach().cpu().item()

        return loss_total

    def save_model(self):
        pass

    def load_model(self):
        pass
