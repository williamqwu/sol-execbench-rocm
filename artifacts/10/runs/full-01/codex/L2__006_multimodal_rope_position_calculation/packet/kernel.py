import torch
import triton
import triton.language as tl


@triton.jit
def _mrope_kernel(
    image_grid,
    video_grid,
    seconds,
    position_ids,
    deltas,
    SEQ: tl.constexpr,
    BATCH: tl.constexpr,
    NUM_IMAGES: tl.constexpr,
    NUM_VIDEOS: tl.constexpr,
    TILE: tl.constexpr,
    MAX_IMAGES: tl.constexpr,
    MAX_VIDEOS: tl.constexpr,
):
    tile_id = tl.program_id(0)
    batch = tl.program_id(1)
    offsets = tile_id * TILE + tl.arange(0, TILE)
    valid = offsets < SEQ

    images_q: tl.constexpr = NUM_IMAGES // BATCH
    images_r: tl.constexpr = NUM_IMAGES % BATCH
    videos_q: tl.constexpr = NUM_VIDEOS // BATCH
    videos_r: tl.constexpr = NUM_VIDEOS % BATCH
    num_images = images_q + (batch < images_r)
    num_videos = videos_q + (batch < videos_r)
    image_index = batch * images_q + tl.minimum(batch, images_r)
    video_index = batch * videos_q + tl.minimum(batch, videos_r)

    out_t = offsets
    out_h = offsets
    out_w = offsets
    source_start = 0
    position_start = 0
    # get_inputs starts each batch at input position 5: vision-start is at 5
    # and the modality token (the reference's first grid position) is at 6.
    media_at = 6

    for item in range(MAX_IMAGES):
        active = item < num_images
        grid_index = image_index + item
        grid_t = tl.load(image_grid + grid_index * 3, mask=active, other=1).to(tl.int32)
        grid_h = (tl.load(image_grid + grid_index * 3 + 1,
                          mask=active, other=2) // 2).to(tl.int32)
        grid_w = (tl.load(image_grid + grid_index * 3 + 2,
                          mask=active, other=2) // 2).to(tl.int32)
        grid_size = grid_t * grid_h * grid_w
        grid_base = position_start + media_at - source_start

        in_grid = active & valid & (offsets >= media_at) & (offsets < media_at + grid_size)
        linear = offsets - media_at
        plane = grid_h * grid_w
        spatial = linear % plane
        height = spatial // grid_w
        width = spatial % grid_w
        out_t = tl.where(in_grid, grid_base, out_t)
        out_h = tl.where(in_grid, grid_base + height, out_h)
        out_w = tl.where(in_grid, grid_base + width, out_w)

        grid_max = tl.maximum(grid_h - 1, grid_w - 1)
        next_source = media_at + grid_size
        next_position = grid_base + grid_max + 1
        after_grid = active & valid & (offsets >= next_source)
        shifted = offsets + next_position - next_source
        out_t = tl.where(after_grid, shifted, out_t)
        out_h = tl.where(after_grid, shifted, out_h)
        out_w = tl.where(after_grid, shifted, out_w)
        source_start = tl.where(active, next_source, source_start)
        position_start = tl.where(active, next_position, position_start)
        media_at = tl.where(active, media_at + grid_size + 5, media_at)

    for item in range(MAX_VIDEOS):
        active = item < num_videos
        grid_index = video_index + item
        grid_t = tl.load(video_grid + grid_index * 3, mask=active, other=1).to(tl.int32)
        grid_h = (tl.load(video_grid + grid_index * 3 + 1,
                          mask=active, other=2) // 2).to(tl.int32)
        grid_w = (tl.load(video_grid + grid_index * 3 + 2,
                          mask=active, other=2) // 2).to(tl.int32)
        second = tl.load(seconds + grid_index, mask=active, other=0.0)
        grid_size = grid_t * grid_h * grid_w
        grid_base = position_start + media_at - source_start

        in_grid = active & valid & (offsets >= media_at) & (offsets < media_at + grid_size)
        linear = offsets - media_at
        plane = grid_h * grid_w
        temporal = linear // plane
        spatial = linear % plane
        height = spatial // grid_w
        width = spatial % grid_w
        temporal_position = (temporal.to(tl.float32) * second * 2.0).to(tl.int32)
        out_t = tl.where(in_grid, grid_base + temporal_position, out_t)
        out_h = tl.where(in_grid, grid_base + height, out_h)
        out_w = tl.where(in_grid, grid_base + width, out_w)

        last_temporal = ((grid_t - 1).to(tl.float32) * second * 2.0).to(tl.int32)
        grid_max = tl.maximum(last_temporal, tl.maximum(grid_h - 1, grid_w - 1))
        next_source = media_at + grid_size
        next_position = grid_base + grid_max + 1
        after_grid = active & valid & (offsets >= next_source)
        shifted = offsets + next_position - next_source
        out_t = tl.where(after_grid, shifted, out_t)
        out_h = tl.where(after_grid, shifted, out_h)
        out_w = tl.where(after_grid, shifted, out_w)
        source_start = tl.where(active, next_source, source_start)
        position_start = tl.where(active, next_position, position_start)
        media_at = tl.where(active, media_at + grid_size + 5, media_at)

    row = batch * SEQ + offsets
    tl.store(position_ids + row, out_t, mask=valid)
    tl.store(position_ids + BATCH * SEQ + row, out_h, mask=valid)
    tl.store(position_ids + 2 * BATCH * SEQ + row, out_w, mask=valid)
    if tile_id == 0:
        tl.store(deltas + batch, position_start - source_start)


def run(input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts,
        attention_mask):
    batch, seq = input_ids.shape
    num_images = image_grid_thw.shape[0]
    num_videos = video_grid_thw.shape[0]
    position_ids = torch.empty((3, batch, seq), dtype=torch.int64,
                               device=input_ids.device)
    deltas = torch.empty((batch, 1), dtype=torch.int64,
                         device=input_ids.device)
    tile = 256
    max_images = (num_images + batch - 1) // batch
    max_videos = (num_videos + batch - 1) // batch
    _mrope_kernel[(triton.cdiv(seq, tile), batch)](
        image_grid_thw, video_grid_thw, second_per_grid_ts,
        position_ids, deltas,
        SEQ=seq, BATCH=batch, NUM_IMAGES=num_images, NUM_VIDEOS=num_videos,
        TILE=tile, MAX_IMAGES=max_images, MAX_VIDEOS=max_videos,
        num_warps=4,
    )
    return position_ids, deltas
