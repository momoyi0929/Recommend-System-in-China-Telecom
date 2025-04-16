import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self,user_num_features, item_num_features,
                 user_categ_sizes, item_categ_sizes, embedding_dim=128):
        super().__init__()

        # 替换Embedding为动态生成的MLP
        self.user_id_to_embedding = nn.Sequential(
            nn.Linear(1, 1024),  # 输入是用户ID（标量），升维到256
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 512),  # 输入是用户ID（标量），升维到256
            nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(0.3),

            nn.Linear(512, 512),  # 输入是用户ID（标量），升维到256
            nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(0.3),

            nn.Linear(512, 256),  # 输入是用户ID（标量），升维到256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, embedding_dim)  # 输出用户Embedding
        )

        self.item_id_to_embedding = nn.Sequential(
            nn.Linear(1, 1024),  # 输入是用户ID（标量），升维到256
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 512),  # 输入是用户ID（标量），升维到256
            nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(0.3),

            nn.Linear(512, 512),  # 输入是用户ID（标量），升维到256
            nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(0.3),

            nn.Linear(512, 256),  # 输入是用户ID（标量），升维到256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, embedding_dim)  # 输出用户Embedding
        )

        # ✅ 用户类别特征 Embedding
        self.user_categ_embeddings = nn.ModuleList(
            [nn.Embedding(size, embedding_dim) for size in user_categ_sizes]
        )

        # ✅ 商品类别特征 Embedding
        self.item_categ_embeddings = nn.ModuleList(
            [nn.Embedding(size, embedding_dim) for size in item_categ_sizes]
        )


        # ✅ 用户塔（User Tower）
        self.user_fc = nn.Sequential(
            nn.Linear(user_num_features + len(user_categ_sizes) * embedding_dim + embedding_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1),

            # 隐藏层1
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Dropout(0.3),

            # 隐藏层1
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(0.3),

            # 隐藏层1
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(0.3),

            # 隐藏层1
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            # # 新增的隐藏层
            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Dropout(0.3),

            nn.Linear(512, embedding_dim)
        )

        # ✅ 商品塔（Item Tower）
        self.item_fc = nn.Sequential(
            nn.Linear(item_num_features + len(item_categ_sizes) * embedding_dim + embedding_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1),

            # 新增的隐藏层
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            # nn.ReLU(),
            # nn.Dropout(0.3),

            # 新增的隐藏层
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(0.3),

            # 新增的隐藏层
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(0.3),

            # 隐藏层1
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            #
            # # 新增的隐藏层
            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Dropout(0.3),

            nn.Linear(512, embedding_dim)
        )

        # ✅ **Match MLP** 计算匹配分数
        self.match_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, 1024),  # 连接 u, v, u*v
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出匹配概率
        )

    def forward(self, user_id, item_id, user_features, user_categ, item_features, item_categ):

        # 动态生成用户/物品Embedding（替代原Embedding查表）
        user_emb = self.user_id_to_embedding(user_id.float().unsqueeze(-1))  # 输入形状：[batch_size, 1]
        item_emb = self.item_id_to_embedding(item_id.float().unsqueeze(-1))

        # ✅ 计算用户类别特征的 Embedding，并拼接
        user_categ_embeds = [emb(user_categ[:, i]) for i, emb in enumerate(self.user_categ_embeddings)]
        user_categ_embeds = torch.cat(user_categ_embeds, dim=1)

        # ✅ 计算商品类别特征的 Embedding，并拼接
        item_categ_embeds = [emb(item_categ[:, i]) for i, emb in enumerate(self.item_categ_embeddings)]
        item_categ_embeds = torch.cat(item_categ_embeds, dim=1)

        # ✅ 拼接所有用户信息
        user_input = torch.cat([user_emb, user_categ_embeds, user_features], dim=1)
        user_vector = self.user_fc(user_input)

        # ✅ 拼接所有商品信息
        item_input = torch.cat([item_emb, item_categ_embeds, item_features], dim=1)
        item_vector = self.item_fc(item_input)

        interaction_vector = torch.cat([user_vector, item_vector, user_vector * item_vector], dim=1)
        final_output = self.match_mlp(interaction_vector)

        # ✅ 返回最终输出
        return final_output


class TwoTowerModel0(nn.Module):
    def __init__(self,user_num_features, item_num_features,
                 user_categ_sizes, item_categ_sizes, embedding_dim=128,
                 device = "mps" if torch.backends.mps.is_available() else "cpu"):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.device = device

        # 替换Embedding为动态生成的MLP
        self.user_id_to_embedding = nn.Sequential(
            nn.Linear(1, 512),  # 输入是用户ID（标量），升维到256
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),  # 输入是用户ID（标量），升维到256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 256),  # 输入是用户ID（标量），升维到256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, embedding_dim)  # 输出用户Embedding
        )

        self.item_id_to_embedding = nn.Sequential(
            nn.Linear(1, 512),  # 输入是用户ID（标量），升维到256
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),  # 输入是用户ID（标量），升维到256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 256),  # 输入是用户ID（标量），升维到256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, embedding_dim)  # 输出用户Embedding
        )

        # ✅ 用户类别特征 Embedding
        self.user_categ_embeddings = nn.ModuleList(
            [nn.Embedding(size, embedding_dim) for size in user_categ_sizes]
        )

        # ✅ 商品类别特征 Embedding
        self.item_categ_embeddings = nn.ModuleList(
            [nn.Embedding(size, embedding_dim) for size in item_categ_sizes]
        )


        # ✅ 用户塔（User Tower）
        self.user_fc = nn.Sequential(
            nn.Linear(571, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # 隐藏层1
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # 隐藏层1
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # 隐藏层1
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, embedding_dim)
        )

        # ✅ 商品塔（Item Tower）
        self.item_fc = nn.Sequential(
            nn.Linear(513, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # 新增的隐藏层
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # 新增的隐藏层
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # 新增的隐藏层
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, embedding_dim)
        )

        # ✅ **Match MLP** 计算匹配分数
        self.match_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, 512),  # 连接 u, v, u*v
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出匹配概率
        )

        # 注意力权重层（可学习）
        self.user_attn_weights = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出 attention score
        ).to(self.device)


        # 注意力权重层（可学习）
        self.item_attn_weights = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出 attention score
        ).to(self.device)

    def forward(self, user_id, item_id, user_features, user_categ, item_features, item_categ):

        # # 动态生成用户/物品Embedding（替代原Embedding查表）
        user_emb = self.user_id_to_embedding(user_id.float().unsqueeze(-1))  # 输入形状：[batch_size, 1]
        item_emb = self.item_id_to_embedding(item_id.float().unsqueeze(-1))

        # 每个类别特征 Embedding 的 shape: [batch, embed_dim]
        user_categ_embeds = [emb(user_categ[:, i]) for i, emb in enumerate(self.user_categ_embeddings)]
        user_categ_embeds = torch.stack(user_categ_embeds, dim=1)  # [batch, num_categ, embed_dim]

        # 计算注意力权重
        user_attn_scores = self.user_attn_weights(user_categ_embeds)  # [batch, num_categ, 1]
        user_attn_weights = torch.softmax(user_attn_scores, dim=1)  # [batch, num_categ, 1]

        # 加权求和
        user_categ_attn_embed = torch.sum(user_attn_weights * user_categ_embeds, dim=1)  # [batch, embed_dim]
        # ----------------------------
        # 每个类别特征 Embedding 的 shape: [batch, embed_dim]
        item_categ_embeds = [emb(item_categ[:, i]) for i, emb in enumerate(self.item_categ_embeddings)]
        item_categ_embeds = torch.stack(item_categ_embeds, dim=1)  # [batch, num_categ, embed_dim]

        # 计算注意力权重
        item_attn_scores = self.item_attn_weights(item_categ_embeds)  # [batch, num_categ, 1]
        item_attn_weights = torch.softmax(item_attn_scores, dim=1)  # [batch, num_categ, 1]

        # 加权求和
        item_categ_attn_embed = torch.sum(item_attn_weights * item_categ_embeds, dim=1)  # [batch, embed_dim]

        # ✅ 拼接所有用户信息
        user_input = torch.cat([user_emb, user_categ_attn_embed, user_features], dim=1)
        user_vector = self.user_fc(user_input)

        # ✅ 拼接所有商品信息
        item_input = torch.cat([item_emb, item_categ_attn_embed, item_features], dim=1)
        item_vector = self.item_fc(item_input)

        interaction_vector = torch.cat([user_vector, item_vector, user_vector * item_vector], dim=1)
        final_output = self.match_mlp(interaction_vector)

        # ✅ 返回最终输出
        return final_output


