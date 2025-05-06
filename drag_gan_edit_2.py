import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T

def drag_gan_edit(
    generator,
    mapping_network,
    initial_latent,
    source_points,
    target_points,
    num_steps=1000,
    lr=1e-3,
    device="cuda",
    is_w_input=False,
    log_resolution=7,
    stopping_threshold=2.0
):
    if is_w_input:
        w = initial_latent.detach().clone().requires_grad_(True)
    else:
        w = mapping_network(initial_latent).detach().clone().requires_grad_(True)

    optimizer = optim.Adam([w], lr=lr)

    # Pretrained VGG feature extractor (up to relu2_2)
    vgg = models.vgg16(pretrained=True).features[:9].to(device).eval()
    for p in vgg.parameters():
        p.requires_grad = False

    transform = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    # Fixed noise
    fixed_noise = []
    resolution = 4
    for i in range(log_resolution):
        n1 = torch.randn(1, 1, resolution, resolution).to(device) if i != 0 else None
        n2 = torch.randn(1, 1, resolution, resolution).to(device)
        fixed_noise.append((n1, n2))
        resolution *= 2

    # Start point tracking
    current_points = source_points.copy()

    for step in range(num_steps):
        optimizer.zero_grad()

        w_broadcast = [w for _ in range(log_resolution)]
        img = generator(w_broadcast, fixed_noise)[0]
        img = (img * 0.5 + 0.5).clamp(0, 1)

        # Resize image to 224x224 for VGG
        img_224 = nn.functional.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        img_norm = transform(img_224.squeeze(0)).unsqueeze(0)
        feat = vgg(img_norm)  # Output shape: (1, C, Hf, Wf)

        feat_h, feat_w = feat.shape[2], feat.shape[3]
        img_h, img_w = img.shape[1], img.shape[2]

        loss = 0.0
        new_points = []

        for (x_s, y_s), (x_t, y_t) in zip(current_points, target_points):
            # Convert image coords → feature map coords
            xs = int(x_s / img_w * feat_w)
            ys = int(y_s / img_h * feat_h)
            xt = int(x_t / img_w * feat_w)
            yt = int(y_t / img_h * feat_h)

            if all(0 <= v < feat_w for v in [xs, xt]) and all(0 <= v < feat_h for v in [ys, yt]):
                src_feat = feat[:, :, ys, xs]
                tgt_feat = feat[:, :, yt, xt]
                loss += torch.norm(src_feat - tgt_feat, p=2)

                # Gradually update source toward target
                dx = int(0.2 * (x_t - x_s))
                dy = int(0.2 * (y_t - y_s))
                new_x = x_s + dx
                new_y = y_s + dy

                dist = ((x_t - x_s) ** 2 + (y_t - y_s) ** 2) ** 0.5
                if dist < stopping_threshold:
                    new_points.append((x_t, y_t))
                else:
                    new_points.append((new_x, new_y))
            else:
                new_points.append((x_s, y_s))  # fallback

        current_points = new_points

        loss.backward()
        optimizer.step()

    return img.detach().cpu(), w.detach().cpu()
