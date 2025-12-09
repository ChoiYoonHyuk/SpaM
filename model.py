import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB
from torch_geometric.datasets import WikipediaNetwork, Actor, WebKB
from torch_geometric.datasets.heterophilous_graph_dataset import HeterophilousGraphDataset
from torch_geometric.nn import GATConv, GCNConv


def load_dataset(data_id: int):
    if data_id == 0:
        dataset = HeterophilousGraphDataset(root='/tmp/RomanEmpire', name='Roman-empire')
    elif data_id == 1:
        dataset = HeterophilousGraphDataset(root='/tmp/Minesweeper', name='Minesweeper')
    elif data_id == 2:
        dataset = HeterophilousGraphDataset(root='/tmp/AmazonRatings', name='Amazon-ratings')
    elif data_id == 3:
        dataset = WikipediaNetwork(root='/tmp/Chameleon', name='chameleon')
    elif data_id == 4:
        dataset = WikipediaNetwork(root='/tmp/Squirrel', name='squirrel')
    elif data_id == 5:
        dataset = Actor(root='/tmp/Actor')
    elif data_id == 6:
        dataset = WebKB(root='/tmp/Cornell', name='Cornell')
    elif data_id == 7:
        dataset = WebKB(root='/tmp/Texas', name='Texas')
    else:
        dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')
    return dataset, dataset.num_classes

class StructuralEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        edge_hidden_dim=64,
        num_classes=0,
        use_gat=True,
        dropout=0.0,
        use_labels=True,
        prior_probs=None,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_labels = use_labels
        self.num_classes = num_classes
        label_dim = num_classes if (use_labels and num_classes > 0) else 0
        feat_in = in_dim + label_dim
        if use_gat:
            self.conv1 = GATConv(feat_in, hidden_dim, heads=1, concat=True)
            self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=True)
        else:
            self.conv1 = GCNConv(feat_in, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, 3),
        )
        if prior_probs is None:
            prior = torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=torch.float)
        else:
            prior = prior_probs.float()
            prior = prior / prior.sum()
        self.register_buffer("prior_probs", prior)
        self.register_buffer("prior_log_probs", torch.log(prior + 1e-12))

    def forward(self, x, edge_index, y=None, train_mask=None):
        if self.use_labels and self.num_classes > 0 and y is not None and train_mask is not None:
            label_feat = torch.zeros(x.size(0), self.num_classes, device=x.device)
            label_feat[train_mask] = F.one_hot(y[train_mask], num_classes=self.num_classes).float()
            x_in = torch.cat([x, label_feat], dim=-1)
        else:
            x_in = x
        h = self.conv1(x_in, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        row, col = edge_index
        edge_feat = torch.cat([h[row], h[col]], dim=-1)
        edge_logits = self.edge_mlp(edge_feat)
        edge_probs = F.softmax(edge_logits, dim=-1)
        prior_log_probs = self.prior_log_probs.to(edge_probs.device)
        kl_per_edge = (edge_probs * (edge_probs.clamp_min(1e-12).log() - prior_log_probs)).sum(dim=-1)
        kl_mean = kl_per_edge.mean()
        p_edge_exist = edge_probs[:, 0] + edge_probs[:, 2]
        recon_log_prob = p_edge_exist.clamp_min(1e-12).log().mean()
        struct_loss = kl_mean - recon_log_prob
        return edge_logits, edge_probs, struct_loss

class S2Layer(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        val_dim,
        sign_emb_dim=8,
        init_gamma=1.0,
        l1_lambda=0.1,
        dropout=0.0,
        residual=True,
        use_self=True,
    ):
        super().__init__()
        self.l1_lambda = l1_lambda
        self.dropout = dropout
        self.residual = residual
        self.use_self = use_self
        self.W_v = nn.Linear(in_dim, val_dim, bias=False)
        self.W_t = nn.Linear(in_dim, val_dim, bias=False)
        self.sign_emb = nn.Embedding(3, sign_emb_dim)
        self.alpha_mlp = nn.Sequential(
            nn.Linear(2 * val_dim + sign_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        if use_self:
            self.W_self = nn.Linear(in_dim, hidden_dim, bias=False)
        else:
            self.W_self = None
        self.W_out = nn.Linear(val_dim, hidden_dim)
        self.gamma_param = nn.Parameter(torch.tensor(init_gamma))

    def forward(self, H, edge_index, edge_sign):
        n_nodes, _ = H.size()
        row, col = edge_index
        V = self.W_v(H)
        T = self.W_t(H)
        t_i = T[col]
        v_j = V[row]
        sign_idx = (edge_sign.long() + 1).clamp(0, 2)
        s_emb = self.sign_emb(sign_idx)
        alpha_input = torch.cat([t_i, v_j, s_emb], dim=-1)
        alpha = self.alpha_mlp(alpha_input).squeeze(-1)
        alpha = F.softshrink(alpha, lambd=self.l1_lambda)
        sparse_loss = alpha.abs().mean()
        pos_mask = edge_sign > 0
        neg_mask = edge_sign < 0
        pos_agg = torch.zeros(n_nodes, V.size(1), device=H.device)
        neg_agg = torch.zeros(n_nodes, V.size(1), device=H.device)
        if pos_mask.any():
            pos_msg = alpha[pos_mask].unsqueeze(-1) * v_j[pos_mask]
            pos_idx = col[pos_mask]
            pos_agg.index_add_(0, pos_idx, pos_msg)
        if neg_mask.any():
            neg_msg = alpha[neg_mask].abs().unsqueeze(-1) * v_j[neg_mask]
            neg_idx = col[neg_mask]
            neg_agg.index_add_(0, neg_idx, neg_msg)
        gamma_val = F.softplus(self.gamma_param)
        signed_agg = pos_agg - gamma_val * neg_agg
        H_neigh = self.W_out(signed_agg)
        if self.W_self is not None:
            H_self = self.W_self(H)
        else:
            H_self = 0.0
        H_new = H_neigh + H_self
        if self.residual and H_new.shape == H.shape:
            H_new = H_new + H
        H_new = F.dropout(H_new, p=self.dropout, training=self.training)
        return H_new, sparse_loss

class SpaM(nn.Module):
    def __init__(
        self,
        in_feats,
        hidden_dim,
        num_classes,
        num_layers=2,
        K=3,
        val_dim=64,
        sign_emb_dim=8,
        init_gamma=1.0,
        l1_lambda=0.1,
        dropout=0.5,
        use_gat=True,
        prior_probs=None,
    ):
        super().__init__()
        self.K = K
        self.dropout = dropout
        self.struct_encoder = StructuralEncoder(
            in_dim=in_feats,
            hidden_dim=hidden_dim,
            edge_hidden_dim=hidden_dim,
            num_classes=num_classes,
            use_gat=use_gat,
            dropout=dropout,
            use_labels=True,
            prior_probs=prior_probs,
        )
        self.backbone1 = GCNConv(in_feats, hidden_dim)
        self.backbone2 = GCNConv(hidden_dim, hidden_dim)
        self.backbone_proj = nn.Linear(in_feats, hidden_dim, bias=False)
        layers = []
        in_dim_layer = hidden_dim
        for _ in range(num_layers):
            layers.append(
                S2Layer(
                    in_dim=in_dim_layer,
                    hidden_dim=hidden_dim,
                    val_dim=val_dim,
                    sign_emb_dim=sign_emb_dim,
                    init_gamma=init_gamma,
                    l1_lambda=l1_lambda,
                    dropout=dropout,
                    residual=True,
                    use_self=True,
                )
            )
            in_dim_layer = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def _sample_signs(edge_logits, tau=0.5):
        y_hard = F.gumbel_softmax(edge_logits, tau=tau, hard=True)
        sign_values = edge_logits.new_tensor([-1.0, 0.0, 1.0])
        edge_sign = (y_hard * sign_values).sum(dim=-1)
        return edge_sign

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        y, train_mask = data.y, data.train_mask
        h1 = self.backbone1(x, edge_index)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.backbone2(h1, edge_index)
        h2 = h2 + self.backbone_proj(x)
        h2 = F.relu(h2)
        H0 = F.dropout(h2, p=self.dropout, training=self.training)
        edge_logits, edge_probs, struct_loss = self.struct_encoder(x, edge_index, y, train_mask)
        logits_list = []
        sparse_losses = []
        for _ in range(self.K):
            edge_sign = self._sample_signs(edge_logits)
            H = H0
            sparse_loss_sum = 0.0
            for layer in self.layers:
                H, sparse_loss_layer = layer(H, edge_index, edge_sign)
                H = F.relu(H)
                sparse_loss_sum = sparse_loss_sum + sparse_loss_layer
            sparse_loss_k = sparse_loss_sum / len(self.layers)
            logits_k = self.classifier(H)
            logits_list.append(logits_k)
            sparse_losses.append(sparse_loss_k)
        logits_stack = torch.stack(logits_list, dim=0)
        probs_stack = F.softmax(logits_stack, dim=-1)
        probs_mc = probs_stack.mean(dim=0)
        logits_mc = (probs_mc + 1e-12).log()
        if self.training:
            probs_train = probs_mc[train_mask]
            y_train = y[train_mask]
            cls_loss = F.nll_loss((probs_train + 1e-12).log(), y_train)
        else:
            cls_loss = None
        sparse_loss = torch.stack(sparse_losses).mean()
        return {
            "logits": logits_mc,
            "cls_loss": cls_loss,
            "sparse_loss": sparse_loss,
            "struct_loss": struct_loss,
        }

def main():
    parser = argparse.ArgumentParser(description='SpaM Node Classification')
    parser.add_argument('data', type=int, help='data selector')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    data_id = args.data
    dataset, num_class = load_dataset(data_id)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    if data.train_mask.dim() == 2:
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
    prior_probs = torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=torch.float)
    init_gamma = 1.0
    l1_lambda = 0.05
    hidden_dim = 64
    dropout = 0.5
    num_layers = 2
    K = 5
    lambda_sp = 1e-3
    lambda_st = 1e-3
    lr = 1e-3
    patience = 200
    max_epochs = 2000
    model = SpaM(
        in_feats=dataset.num_node_features,
        hidden_dim=hidden_dim,
        num_classes=num_class,
        num_layers=num_layers,
        K=K,
        val_dim=64,
        sign_emb_dim=8,
        init_gamma=init_gamma,
        l1_lambda=l1_lambda,
        dropout=dropout,
        use_gat=True,
        prior_probs=prior_probs,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=50, min_lr=1e-5
    )
    best_val = 0.0
    best_test = 0.0
    patience_ctr = 0
    warmup_epochs = 50
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss_cls = out["cls_loss"]
        loss_sparse = out["sparse_loss"]
        loss_struct = out["struct_loss"]
        if epoch <= warmup_epochs:
            loss = loss_cls
        else:
            loss = loss_cls + lambda_sp * loss_sparse + lambda_st * loss_struct
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            out_eval = model(data)
            logits_eval = out_eval["logits"]
            pred = logits_eval.argmax(dim=1)
            val_acc = pred[data.val_mask].eq(data.y[data.val_mask]).float().mean().item()
            test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).float().mean().item()
            if test_acc > best_test:
                best_test = test_acc
        scheduler.step(val_acc)
        if val_acc > best_val:
            best_val = val_acc
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr > patience:
                print(f"Early stopping at epoch {epoch}")
                break
        if epoch % 50 == 0 or val_acc == best_val:
            print(
                f"Epoch {epoch:04d} | Loss: {loss.item():.4f} | "
                f"L_cls: {loss_cls.item():.4f} | L_sp: {loss_sparse.item():.4f} | "
                f"L_st: {loss_struct.item():.4f} | "
                f"Val: {val_acc:.4f} | Test: {test_acc:.4f} | Best Test: {best_test:.4f}"
            )

if __name__ == '__main__':
    main()
