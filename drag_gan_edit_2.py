import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T

def extract_patch(feat, x, y, size=5):
    x1 = max(x - size, 0)
    y1 = max(y - size, 0)
    x2 = min(x + size + 1, feat.shape[-1])
    y2 = min(y + size + 1, feat.shape[-2])
    return feat[:, :, y1:y2, x1:x2]

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
    stopping_threshold=2.0,
    return_intermediates=False,
    intermediate_interval=200
):
    if is_w_input:
        w = initial_latent.detach().clone().requires_grad_(True)
    else:
        w = mapping_network(initial_latent).detach().clone().requires_grad_(True)

    w_broadcast = []
    for i in range(log_resolution):
        w_clone = w.clone().detach().requires_grad_(i >= log_resolution - 4)
        w_broadcast.append(w_clone)

    optimizer = optim.Adam([wi for wi in w_broadcast if wi.requires_grad], lr=lr)

    # Pretrained VGG feature extractor (up to relu3_3)
    vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
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
    intermediate_images = []
    loss_history = []

    for step in range(num_steps):
        optimizer.zero_grad()

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
                src_patch = extract_patch(feat, xs, ys)
                tgt_patch = extract_patch(feat, xt, yt)
                loss += torch.nn.functional.l1_loss(src_patch, tgt_patch)

                # Gradually update source toward target
                dx = 0.15 * (x_t - x_s)
                dy = 0.15 * (y_t - y_s)
                new_x = int(x_s + dx)
                new_y = int(y_s + dy)

                dist = ((x_t - x_s) ** 2 + (y_t - y_s) ** 2) ** 0.5
                if dist < stopping_threshold:
                    new_points.append((x_t, y_t))
                else:
                    new_points.append((new_x, new_y))
            else:
                new_points.append((x_s, y_s))  # fallback

        current_points = new_points
        loss += 0.002 * sum(torch.norm(wi, p=2) for wi in w_broadcast if wi.requires_grad)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())

        if return_intermediates and step % intermediate_interval == 0:
            with torch.no_grad():
                img_vis = (img * 255).clamp(0, 255).byte()
                intermediate_images.append(img_vis.cpu())

    final_img = img.detach().cpu()
    final_ws = [wi.detach().cpu() for wi in w_broadcast]

    if return_intermediates:
        return final_img, final_ws, intermediate_images, loss_history
    else:
        return final_img, final_ws
