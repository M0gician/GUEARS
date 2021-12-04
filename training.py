import torch
import torch.nn as nn

class Trainer:
    def __init__(self, data, model, user_embedding, item_embedding, optimizer, device, args):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = nn.SmoothL1Loss()
        self.device = device
        self.args = args
        self.user_embedding = user_embedding
        self.item_embedding = item_embedding


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
            attention_mask = torch.ones_like(rewards).squeeze()

            states = states[:, :341].to(self.device)
            actions = actions[:, :341].to(self.device)
            rewards = rewards[:, :341].to(self.device)
            rtg = rtg[:, :341].to(self.device)
            timesteps = timesteps[:, :341].to(self.device)
            attention_mask = attention_mask[:, :341].to(self.device)
            action_target = actions.clone()
            # print(states.shape, actions.shape, rewards.shape, rtg.shape, timesteps.shape, attention_mask.shape)

            # do forward
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rewards, rtg, timesteps, attention_mask=attention_mask,
            )

            # compute loss
            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

            loss = self.loss_fn(action_target, action_preds)

            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            self.optimizer.step()
            loss_total += loss.detach().cpu().item()

        return loss_total

    def get_user_embedding(self, user_id):
        return self.user_embedding(user_id)

    def get_item_embedding(self, item_id):
        return self.item_embedding(item_id)

    def save_model(self):
        pass

    def load_model(self):
        pass
