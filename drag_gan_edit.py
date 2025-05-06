import torch
import torch.optim as optim
import lpips

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
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)

    # Fixed noise for consistent outputs
    fixed_noise = []
    resolution = 4
    for i in range(log_resolution):
        n1 = torch.randn(1, 1, resolution, resolution).to(device) if i != 0 else None
        n2 = torch.randn(1, 1, resolution, resolution).to(device)
        fixed_noise.append((n1, n2))
        resolution *= 2

    # Initial image for reference
    with torch.no_grad():
        w_init = [w.detach() for _ in range(log_resolution)]
        original_img = generator(w_init, fixed_noise)[0]
        original_img = (original_img * 0.5 + 0.5).clamp(0, 1)

    # Current point starts at initial source
    current_points = source_points.copy()

    for step in range(num_steps):
        optimizer.zero_grad()
        w_broadcast = [w for _ in range(log_resolution)]
        img = generator(w_broadcast, fixed_noise)[0]
        img = (img * 0.5 + 0.5).clamp(0, 1)

        loss = 0.0
        new_points = []

        for i, ((x_s, y_s), (x_t, y_t)) in enumerate(zip(current_points, target_points)):
            # Directional constraint: force pixel at current (x_s, y_s) to move toward (x_t, y_t)
            if 0 <= x_s < img.shape[2] and 0 <= y_s < img.shape[1] and \
               0 <= x_t < img.shape[2] and 0 <= y_t < img.shape[1]:

                diff = img[:, y_t, x_t] - img[:, y_s, x_s]
                loss += torch.norm(diff, p=2)

                # Move current source toward target slightly to simulate tracking
                new_x = int(x_s + 0.2 * (x_t - x_s))
                new_y = int(y_s + 0.2 * (y_t - y_s))
                new_points.append((new_x, new_y))

                # Optional: stop if close enough
                if ((x_t - x_s)**2 + (y_t - y_s)**2) ** 0.5 < stopping_threshold:
                    new_points[-1] = (x_t, y_t)
            else:
                new_points.append((x_s, y_s))  # fallback

        current_points = new_points

        # Add LPIPS perceptual similarity to keep realism
        perceptual_loss = lpips_loss_fn(img.unsqueeze(0), original_img.unsqueeze(0))
        loss += 0.8 * perceptual_loss.mean()

        loss.backward()
        optimizer.step()

    return img.detach().cpu(), w.detach().cpu()
