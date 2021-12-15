import torch
import torch.nn as nn
import torch.optim as optim
from models.decision_transformer import DecisionTransformer
from torch.utils.data import DataLoader
from src.datasets.dataloader import UserDataset
from training import ModelTrainer, FCTrainer
from models.FullyConnected import FullyConnected

def train():
    user_dataset = UserDataset(mode='train')
    # prepare args
    state_dim = 512
    action_dim = 512
    user_count = 943
    item_count = 1682
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epochs = 50
    learning_rate = 1e-4
    batch_size = 128

    # prepare data loader
    train_loader = DataLoader(user_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    # setting up model
    model = DecisionTransformer(
        state_dim=state_dim,        # state embedding dimension
        act_dim=action_dim,         # action embedding dimension
        max_length=737,             # max sequence length
        max_ep_len=1000,            # maximum episode length
        hidden_size=512,            # hidden_size of the model
        n_layer=3,                  # number of layers
        n_head=1,                   # number of attention heads
        n_inner=4 * 512,
        activation_function='relu',     # internal activation function
        n_positions=1024,
        resid_pdrop=0.1,
        attmn_pdrop=0.1
    ).to(device)
    user_embedding = nn.Embedding(user_count, state_dim).to(device)
    item_embedding = nn.Embedding(item_count, action_dim).to(device)

    # setting up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # instantiate trainer
    args = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "user_count": user_count,
        "item_count": item_count,
        "model_params": "checkpoints/model.pt",
        "user_embeds_params": "checkpoints/user.pt",
        "item_embeds_params": "checkpoints/item.pt",
        "pretrained": False
    }
    trainer = ModelTrainer(train_loader, model, user_embedding, item_embedding, optimizer, device, args)


    """input_state = torch.randn([64, 333, state_dim])
    input_action = torch.randn([64, 333, action_dim])
    input_rewards = torch.randn([64, 333, 1])
    input_dones = torch.rand([64, 333]).bool()
    input_rtg = torch.randn([64, 334, 1])
    input_timesteps = torch.stack([torch.arange(0, 333) for i in range(64)])
    attention_mask = torch.ones_like(input_dones)
    print(input_timesteps)
    state_pred, action_pred, reward_pred = model.forward(input_state, input_action, input_rewards, input_rtg[:,:-1],
                                                         input_timesteps, attention_mask=attention_mask)
    print(state_pred.shape, action_pred.shape, reward_pred.shape)"""

    for epoch in range(epochs):
        loss = trainer.train_epoch()
        print(f"Epoch: {epoch+1} Loss: {loss}")

    """
    Train fully connected to evaluate our embeddings
    """
    fc_args = {
        "user_count": user_count,
        "item_count": item_count,
        "device": device
    }
    fc_epochs = 10
    fc_lr = 1e-4
    K = 5
    fc_model = FullyConnected(state_dim, item_count).to(device)
    fc_optimizer = optim.Adam(fc_model.parameters(), lr=fc_lr)
    fc_trainer = FCTrainer(fc_model, fc_optimizer, user_embedding, fc_args)
    fc_dataset = UserDataset(mode='train')

    for epoch in range(fc_epochs):
        loss = fc_trainer.train(fc_dataset.get_ratings(load_full=True).to(device))
        print(f"Epoch: {epoch+1} Loss: {loss}")

    """
    Get topK from current user
    """
    topK = fc_trainer.test(UserDataset(mode='test').get_test_ratings(load_full=True).to(device), K=K)
    return topK

def main():
    train()


if __name__ == '__main__':
    main()