import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#################################################################################
# CNN backbone: extracts spectrogram features
#################################################################################
class SpectrogramEncoder(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True, embed_dim=128):
        super(SpectrogramEncoder, self).__init__()

        self.embed_dim = embed_dim

        if model_name == 'resnet18':
            self.cnn = models.resnet18(pretrained=pretrained)
            self.cnn.fc = nn.Identity()
            self.output_dim = 512

        elif model_name == 'mobilenet_v2':
            self.cnn = models.mobilenet_v2(pretrained=pretrained)
            self.cnn.classifier = nn.Identity()
            self.output_dim = 1280

        else:
            raise NotImplementedError
        
        # Freeze CNN backbone
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.spec_projector = nn.Linear(self.output_dim, self.embed_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = self.spec_projector(x)
        return x
    
#################################################################################
# Glottal feature embedding
#################################################################################
class GlottalEmbedding(nn.Module):
    def __init__(self, input_dim=9, embed_dim=128):
        super(GlottalEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return F.relu(self.linear(x))
    
#################################################################################
# Transformer-like block
#################################################################################
class TransformerLikeBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1):
        super(TransformerLikeBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_off=False):
        if attn_off:
            attn_output, attn_weights = x, None
        else:
            attn_output, attn_weights = self.self_attn(x, x, x)

        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x, attn_weights
    
#################################################################################
# Spec + Glottal Model
#################################################################################
class SpecGlottalTransformerModel(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1,
                 model_name='resnet18', pretrained=True):
        super().__init__()
        self.spec_encoder = SpectrogramEncoder(model_name=model_name, pretrained=pretrained, embed_dim=embed_dim)
        self.glottal_encoder = GlottalEmbedding(input_dim=9, embed_dim=embed_dim)
        self.transformer_block = TransformerLikeBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, spec, glottal):
        spec_features = self.spec_encoder(spec) # (B, embed_dim)
        glottal_features = self.glottal_encoder(glottal)

        x = torch.stack([spec_features, glottal_features], dim=1) # (B, 2, embed_dim)
        out, attn_weights = self.transformer_block(x, attn_off=False)

        out_pooled = out.mean(dim=1)
        logits = self.classifier(out_pooled)
        return logits, attn_weights