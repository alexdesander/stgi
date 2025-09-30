@group(0) @binding(0)
var<uniform> window_size: vec2<u32>;

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0)       local_pos: vec2<f32>,
    @location(1)       inst_id  : u32,
    @location(2)       size_px  : vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index)  v_idx   : u32,
    @builtin(instance_index)i_idx   : u32,

    /* ─── per-instance data ───────────────────────────────────────────── */
    @location(0) bounds   : vec4<u32>, // [l, b, r, t]
    @location(1) rotation : vec4<f32>, // [cx, cy, sinθ, cosθ]
) -> VertexOutput {

    /* rectangle extents ------------------------------------------------- */
    let l = f32(bounds.x);
    let b = f32(bounds.y);
    let r = f32(bounds.z);
    let t = f32(bounds.w);
    let w = r - l;
    let h = b - t;

    /* node-centre & rotation ------------------------------------------- */
    let cx   = rotation.x;
    let cy   = rotation.y;
    let sinθ = rotation.z;
    let cosθ = rotation.w;

    /* corner lookup ----------------------------------------------------- */
    let corners  = array<vec2<f32>, 4>(
        vec2(0.0, 0.0), vec2(1.0, 0.0),
        vec2(0.0, 1.0), vec2(1.0, 1.0)
    );
    let indices  = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
    let corner   = corners[indices[v_idx]];

    /* axis-aligned pixel position before rotation ----------------------- */
    let pos_px   = vec2(l + corner.x * w,
                        t + corner.y * h);

    /* rotate around the centre ----------------------------------------- */
    let rel      = pos_px - vec2(cx, cy);
    let rot_px   = vec2(
        rel.x * cosθ - rel.y * sinθ,
        rel.x * sinθ + rel.y * cosθ
    ) + vec2(cx, cy);

    /* NDC conversion ---------------------------------------------------- */
    let win      = vec2<f32>(window_size);
    let ndc      = vec2(
        rot_px.x / win.x * 2.0 - 1.0,
        1.0      - rot_px.y / win.y * 2.0
    );

    /* pack outputs ------------------------------------------------------ */
    var out : VertexOutput;
    out.position  = vec4(ndc, 0.0, 1.0);
    out.local_pos = corner;     // unchanged – still 0‥1 across quad
    out.inst_id   = i_idx;
    out.size_px   = vec2(w, h);
    return out;
}

/* unchanged fragment stage --------------------------------------------- */
fn rand(i: u32) -> f32 {
    return fract(sin(f32(i) * 12.9898) * 43758.5453);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let border_px: f32 = 3.0;

    let x_px         = in.local_pos.x * in.size_px.x;
    let y_px         = in.local_pos.y * in.size_px.y;
    let dist_to_edge = min(
        min(x_px, in.size_px.x - x_px),
        min(y_px, in.size_px.y - y_px)
    );

    if (dist_to_edge > border_px) {
        discard;
    }

    let r = rand(in.inst_id * 3u + 0u);
    let g = rand(in.inst_id * 3u + 1u);
    let b = rand(in.inst_id * 3u + 2u);
    return vec4(r, g, b, 1.0);
}
