#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cmath>
#include <limits>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace py = pybind11;


// ---------- Forward kernel ----------
__global__ void backproject_kernel_cone_beam_source_only(
    const float* __restrict__ projections,
    const float* __restrict__ angles,
    const float* __restrict__ sid,
    const float* __restrict__ sod,
    const float* __restrict__ det_u0_arr,
    const float* __restrict__ det_v0_arr,
    const float* __restrict__ src_x_arr,
    const float* __restrict__ src_y_arr,
    const float* __restrict__ src_z_arr,
    float pixel_size,
    float* __restrict__ volume,
    int A, int U_det, int V_det,
    int D, int H, int W,
    float voxel_size,
    float off_x, float off_y, float off_z
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;

    const float vx = (x - W * 0.5f) * voxel_size + off_x;
    const float vy = (y - H * 0.5f) * voxel_size + off_y;
    const float vz = (z - D * 0.5f) * voxel_size + off_z;

    float acc = 0.0f;

    for (int a = 0; a < A; ++a) {
        const float beta = angles[a];
        const float c = cosf(beta);
        const float s = sinf(beta);

        // Object point in the rotating detector coordinate system.
        const float px = vx * c + vy * s;
        const float py = -vx * s + vy * c;
        const float pz = vz;

        // World-coordinate source offset in the same rotating system.
        const float ds_u = src_x_arr[a] * c + src_y_arr[a] * s;
        const float ds_n = -src_x_arr[a] * s + src_y_arr[a] * c;
        const float ds_v = src_z_arr[a];

        // The detector remains at y = sod - sid; only the source moves.
        const float denom = sod[a] + ds_n - py;
        const float source_detector_n = sid[a] + ds_n;
        if (denom <= 1e-6f || source_detector_n <= 1e-6f) continue;

        const float t = source_detector_n / denom;
        const float u_phys = ds_u + t * (px - ds_u);
        const float v_phys = ds_v + t * (pz - ds_v);

        const float source_iso_n = sod[a] + ds_n;
        const float scale_weight = source_iso_n / denom;
        const float geom_w = scale_weight * scale_weight;

        const float u_idx = u_phys / pixel_size + det_u0_arr[a];
        const float v_idx = v_phys / pixel_size + det_v0_arr[a];

        if (u_idx >= 0.0f && u_idx < (U_det - 1.0f) &&
            v_idx >= 0.0f && v_idx < (V_det - 1.0f)) {
            const int u0 = (int)floorf(u_idx);
            const int v0 = (int)floorf(v_idx);
            const float fu = u_idx - u0;
            const float fv = v_idx - v0;

            const float* img = projections + a * U_det * V_det;
            const float p00 = img[u0 * V_det + v0];
            const float p10 = img[(u0 + 1) * V_det + v0];
            const float p01 = img[u0 * V_det + (v0 + 1)];
            const float p11 = img[(u0 + 1) * V_det + (v0 + 1)];

            const float interp =
                (1.0f - fu) * (1.0f - fv) * p00 +
                fu          * (1.0f - fv) * p10 +
                (1.0f - fu) * fv          * p01 +
                fu          * fv          * p11;

            acc += geom_w * interp;
        }
    }

    volume[z * H * W + y * W + x] = acc;
}

// ---------- Backward kernel ----------
__global__ void backproject_backward_kernel_cone_beam_source_only(
    const float* __restrict__ grad_vol,
    const float* __restrict__ proj,
    const float* __restrict__ angles,
    const float* __restrict__ sid,
    const float* __restrict__ sod,
    const float* __restrict__ det_u0_arr,
    const float* __restrict__ det_v0_arr,
    const float* __restrict__ src_x_arr,
    const float* __restrict__ src_y_arr,
    const float* __restrict__ src_z_arr,
    float pixel_size,
    float* __restrict__ grad_proj,
    float* __restrict__ grad_angles,
    float* __restrict__ grad_sid,
    float* __restrict__ grad_sod,
    float* __restrict__ grad_det_u0,
    float* __restrict__ grad_det_v0,
    float* __restrict__ grad_src_x,
    float* __restrict__ grad_src_y,
    float* __restrict__ grad_src_z,
    int A, int U_det, int V_det,
    int D, int H, int W,
    float voxel_size,
    float off_x, float off_y, float off_z
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;

    const float vx = (x - W * 0.5f) * voxel_size + off_x;
    const float vy = (y - H * 0.5f) * voxel_size + off_y;
    const float vz = (z - D * 0.5f) * voxel_size + off_z;
    const float grad_val = grad_vol[z * H * W + y * W + x];
    const float inv_pixel = 1.0f / pixel_size;

    for (int a = 0; a < A; ++a) {
        const float beta = angles[a];
        const float c = cosf(beta);
        const float s = sinf(beta);

        const float px = vx * c + vy * s;
        const float py = -vx * s + vy * c;
        const float pz = vz;

        const float src_x = src_x_arr[a];
        const float src_y = src_y_arr[a];
        const float ds_u = src_x * c + src_y * s;
        const float ds_n = -src_x * s + src_y * c;
        const float ds_v = src_z_arr[a];

        const float sod_a = sod[a];
        const float sid_a = sid[a];
        const float denom = sod_a + ds_n - py;       // q
        const float source_detector_n = sid_a + ds_n; // l
        if (denom <= 1e-6f || source_detector_n <= 1e-6f) continue;

        const float inv_q = 1.0f / denom;
        const float inv_q2 = inv_q * inv_q;
        const float inv_q3 = inv_q2 * inv_q;
        const float t = source_detector_n * inv_q;
        const float du_ray = px - ds_u;
        const float dv_ray = pz - ds_v;

        const float u_phys = ds_u + t * du_ray;
        const float v_phys = ds_v + t * dv_ray;

        const float source_iso_n = sod_a + ds_n; // m
        const float m2 = source_iso_n * source_iso_n;
        const float geom_w = m2 * inv_q2;

        const float u_idx = u_phys * inv_pixel + det_u0_arr[a];
        const float v_idx = v_phys * inv_pixel + det_v0_arr[a];

        if (u_idx >= 0.0f && u_idx < (U_det - 1.0f) &&
            v_idx >= 0.0f && v_idx < (V_det - 1.0f)) {
            const int u0 = (int)floorf(u_idx);
            const int v0 = (int)floorf(v_idx);
            const float fu = u_idx - u0;
            const float fv = v_idx - v0;

            const float* img = proj + a * U_det * V_det;
            const float p00 = img[u0 * V_det + v0];
            const float p10 = img[(u0 + 1) * V_det + v0];
            const float p01 = img[u0 * V_det + (v0 + 1)];
            const float p11 = img[(u0 + 1) * V_det + (v0 + 1)];

            const float interp =
                (1.0f - fu) * (1.0f - fv) * p00 +
                fu          * (1.0f - fv) * p10 +
                (1.0f - fu) * fv          * p01 +
                fu          * fv          * p11;

            // Derivatives of bilinear interpolation with respect to pixel index.
            const float I_u = (1.0f - fv) * (p10 - p00) + fv * (p11 - p01);
            const float I_v = (1.0f - fu) * (p01 - p00) + fu * (p11 - p10);
            const float g_proj_base = grad_val * geom_w;

            float* grad_img = grad_proj + a * U_det * V_det;
            atomicAdd(&grad_img[u0 * V_det + v0],
                      g_proj_base * (1.0f - fu) * (1.0f - fv));
            atomicAdd(&grad_img[(u0 + 1) * V_det + v0],
                      g_proj_base * fu * (1.0f - fv));
            atomicAdd(&grad_img[u0 * V_det + (v0 + 1)],
                      g_proj_base * (1.0f - fu) * fv);
            atomicAdd(&grad_img[(u0 + 1) * V_det + (v0 + 1)],
                      g_proj_base * fu * fv);

            // A small local expression for d[geom_w * interp].
            // dU/dp and dV/dp below are in physical detector units.

            // sid: source stays fixed; the detector plane moves.
            const float dt_dsid = inv_q;
            const float dU_dsid = du_ray * dt_dsid;
            const float dV_dsid = dv_ray * dt_dsid;
            const float dF_dsid = grad_val * geom_w *
                (I_u * dU_dsid * inv_pixel + I_v * dV_dsid * inv_pixel);
            atomicAdd(&grad_sid[a], dF_dsid);

            // sod: nominal source and detector translate together radially.
            const float dt_dsod = -source_detector_n * inv_q2;
            const float dU_dsod = du_ray * dt_dsod;
            const float dV_dsod = dv_ray * dt_dsod;
            const float dW_dsod =
                2.0f * source_iso_n * inv_q2 - 2.0f * m2 * inv_q3;
            const float dF_dsod = grad_val *
                (dW_dsod * interp + geom_w *
                 (I_u * dU_dsod * inv_pixel + I_v * dV_dsod * inv_pixel));
            atomicAdd(&grad_sod[a], dF_dsod);

            // beta.  Both the point and the world-coordinate source offset rotate.
            const float dpx_dbeta = py;
            const float dpy_dbeta = -px;
            const float dsu_dbeta = ds_n;
            const float dsn_dbeta = -ds_u;
            const float ddenom_dbeta = dsn_dbeta - dpy_dbeta; // px - ds_u
            const float dl_dbeta = dsn_dbeta;
            const float dt_dbeta =
                (dl_dbeta * denom - source_detector_n * ddenom_dbeta) * inv_q2;
            const float dU_dbeta =
                (1.0f - t) * dsu_dbeta + t * dpx_dbeta + du_ray * dt_dbeta;
            const float dV_dbeta = dv_ray * dt_dbeta;
            const float dW_dbeta =
                2.0f * source_iso_n * inv_q2 * dsn_dbeta -
                2.0f * m2 * inv_q3 * ddenom_dbeta;
            const float dF_dbeta = grad_val *
                (dW_dbeta * interp + geom_w *
                 (I_u * dU_dbeta * inv_pixel + I_v * dV_dbeta * inv_pixel));
            atomicAdd(&grad_angles[a], dF_dbeta);

            atomicAdd(&grad_det_u0[a], grad_val * geom_w * I_u);
            atomicAdd(&grad_det_v0[a], grad_val * geom_w * I_v);

            // src_x in world coordinates.
            const float dsu_dsrc_x = c;
            const float dsn_dsrc_x = -s;
            const float dt_dsrc_x =
                dsn_dsrc_x * (denom - source_detector_n) * inv_q2;
            const float dU_dsrc_x =
                (1.0f - t) * dsu_dsrc_x + du_ray * dt_dsrc_x;
            const float dV_dsrc_x = dv_ray * dt_dsrc_x;
            const float dW_dsrc_x =
                2.0f * source_iso_n * inv_q2 * dsn_dsrc_x -
                2.0f * m2 * inv_q3 * dsn_dsrc_x;
            const float dF_dsrc_x = grad_val *
                (dW_dsrc_x * interp + geom_w *
                 (I_u * dU_dsrc_x * inv_pixel + I_v * dV_dsrc_x * inv_pixel));
            atomicAdd(&grad_src_x[a], dF_dsrc_x);

            // src_y in world coordinates.
            const float dsu_dsrc_y = s;
            const float dsn_dsrc_y = c;
            const float dt_dsrc_y =
                dsn_dsrc_y * (denom - source_detector_n) * inv_q2;
            const float dU_dsrc_y =
                (1.0f - t) * dsu_dsrc_y + du_ray * dt_dsrc_y;
            const float dV_dsrc_y = dv_ray * dt_dsrc_y;
            const float dW_dsrc_y =
                2.0f * source_iso_n * inv_q2 * dsn_dsrc_y -
                2.0f * m2 * inv_q3 * dsn_dsrc_y;
            const float dF_dsrc_y = grad_val *
                (dW_dsrc_y * interp + geom_w *
                 (I_u * dU_dsrc_y * inv_pixel + I_v * dV_dsrc_y * inv_pixel));
            atomicAdd(&grad_src_y[a], dF_dsrc_y);

            // src_z in world coordinates; the fixed detector gives (1 - t),
            // rather than the old synchronous-motion derivative -t.
            const float dV_dsrc_z = 1.0f - t;
            const float dF_dsrc_z =
                grad_val * geom_w * I_v * dV_dsrc_z * inv_pixel;
            atomicAdd(&grad_src_z[a], dF_dsrc_z);
        }
    }
}

// Packaging-only host checks. The two mathematical CUDA kernels are unchanged.
static void check_projection(const torch::Tensor& proj) {
    TORCH_CHECK(proj.is_cuda() && proj.scalar_type() == torch::kFloat32,
                "proj must be a CUDA float32 tensor");
    TORCH_CHECK(proj.dim() == 3 && proj.size(0) > 0 &&
                proj.size(1) >= 2 && proj.size(2) >= 2,
                "proj must have shape [A,U,V] with A >= 1 and U,V >= 2");
    TORCH_CHECK(proj.numel() <= std::numeric_limits<int>::max(),
                "proj exceeds the kernel's 32-bit indexing limit");
}

static void check_geometry(const torch::Tensor& proj, const torch::Tensor& tensor,
                           const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kFloat32 &&
                tensor.device() == proj.device() && tensor.dim() == 1 &&
                tensor.numel() == proj.size(0),
                name, " must be CUDA float32 on proj.device with shape [A]");
}

static void check_geometry_inputs(
    const torch::Tensor& proj, const torch::Tensor& angles,
    const torch::Tensor& sid, const torch::Tensor& sod,
    const torch::Tensor& det_u0, const torch::Tensor& det_v0,
    const torch::Tensor& src_x, const torch::Tensor& src_y, const torch::Tensor& src_z
) {
    check_projection(proj);
    check_geometry(proj, angles, "angles");
    check_geometry(proj, sid, "sid");
    check_geometry(proj, sod, "sod");
    check_geometry(proj, det_u0, "det_u0");
    check_geometry(proj, det_v0, "det_v0");
    check_geometry(proj, src_x, "src_x");
    check_geometry(proj, src_y, "src_y");
    check_geometry(proj, src_z, "src_z");
}

static void check_volume_shape(int64_t D, int64_t H, int64_t W) {
    const int64_t limit = std::numeric_limits<int>::max();
    TORCH_CHECK(D > 0 && H > 0 && W > 0 && D <= limit && H <= limit && W <= limit,
                "D,H,W must be positive and fit in a 32-bit integer");
    TORCH_CHECK(D <= limit / H && D * H <= limit / W,
                "volume exceeds the kernel's 32-bit indexing limit");
    TORCH_CHECK(D <= 8 * 65535 && H <= 8 * 65535,
                "D,H exceed the kernel's 3D CUDA grid limits");
}

// ---------- Forward wrapper ----------
torch::Tensor forward_cuda(
    torch::Tensor proj,
    torch::Tensor angles,
    torch::Tensor sid,
    torch::Tensor sod,
    torch::Tensor det_u0_arr,
    torch::Tensor det_v0_arr,
    torch::Tensor src_x_arr,
    torch::Tensor src_y_arr,
    torch::Tensor src_z_arr,
    float pixel_size,
    int D, int H, int W,
    float voxel_size,
    float off_x = 0.0f,
    float off_y = 0.0f,
    float off_z = 0.0f
) {
    check_geometry_inputs(proj, angles, sid, sod, det_u0_arr, det_v0_arr,
                          src_x_arr, src_y_arr, src_z_arr);
    check_volume_shape(D, H, W);
    TORCH_CHECK(std::isfinite(pixel_size) && std::isfinite(voxel_size) &&
                std::isfinite(off_x) && std::isfinite(off_y) && std::isfinite(off_z),
                "sizes and ROI offsets must be finite");
    const c10::cuda::CUDAGuard device_guard(proj.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(proj.get_device());
    TORCH_CHECK(proj.is_cuda() && angles.is_cuda() && sid.is_cuda() && sod.is_cuda() &&
                det_u0_arr.is_cuda() && det_v0_arr.is_cuda() &&
                src_x_arr.is_cuda() && src_y_arr.is_cuda() && src_z_arr.is_cuda(),
                "All inputs must be CUDA tensors");
    TORCH_CHECK(proj.scalar_type() == torch::kFloat32 &&
                angles.scalar_type() == torch::kFloat32 &&
                sid.scalar_type() == torch::kFloat32 && sod.scalar_type() == torch::kFloat32 &&
                det_u0_arr.scalar_type() == torch::kFloat32 &&
                det_v0_arr.scalar_type() == torch::kFloat32 &&
                src_x_arr.scalar_type() == torch::kFloat32 &&
                src_y_arr.scalar_type() == torch::kFloat32 &&
                src_z_arr.scalar_type() == torch::kFloat32,
                "All inputs must be float32");
    TORCH_CHECK(pixel_size > 0.0f && voxel_size > 0.0f,
                "pixel_size and voxel_size must be positive");

    auto proj_c = proj.contiguous();
    auto ang_c = angles.contiguous();
    auto sid_c = sid.contiguous();
    auto sod_c = sod.contiguous();
    auto det_u0_c = det_u0_arr.contiguous();
    auto det_v0_c = det_v0_arr.contiguous();
    auto src_x_c = src_x_arr.contiguous();
    auto src_y_c = src_y_arr.contiguous();
    auto src_z_c = src_z_arr.contiguous();

    const int A = (int)proj_c.size(0);
    const int U_det = (int)proj_c.size(1);
    const int V_det = (int)proj_c.size(2);
    TORCH_CHECK(ang_c.numel() == A && sid_c.numel() == A && sod_c.numel() == A &&
                det_u0_c.numel() == A && det_v0_c.numel() == A &&
                src_x_c.numel() == A && src_y_c.numel() == A && src_z_c.numel() == A,
                "Every geometry array must contain A elements");

    auto vol = torch::zeros({D, H, W}, proj_c.options());

    const dim3 threads(8, 8, 8);
    const dim3 blocks((W + threads.x - 1) / threads.x,
                      (H + threads.y - 1) / threads.y,
                      (D + threads.z - 1) / threads.z);

    backproject_kernel_cone_beam_source_only<<<blocks, threads, 0, stream>>>(
        proj_c.data_ptr<float>(), ang_c.data_ptr<float>(),
        sid_c.data_ptr<float>(), sod_c.data_ptr<float>(),
        det_u0_c.data_ptr<float>(), det_v0_c.data_ptr<float>(),
        src_x_c.data_ptr<float>(), src_y_c.data_ptr<float>(), src_z_c.data_ptr<float>(),
        pixel_size, vol.data_ptr<float>(),
        A, U_det, V_det, D, H, W, voxel_size, off_x, off_y, off_z);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return vol;
}

// ---------- Backward wrapper ----------
std::vector<torch::Tensor> backward_cuda(
    torch::Tensor grad_vol,
    torch::Tensor proj,
    torch::Tensor angles,
    torch::Tensor sid,
    torch::Tensor sod,
    torch::Tensor det_u0_arr,
    torch::Tensor det_v0_arr,
    torch::Tensor src_x_arr,
    torch::Tensor src_y_arr,
    torch::Tensor src_z_arr,
    float pixel_size,
    float voxel_size,
    float off_x = 0.0f,
    float off_y = 0.0f,
    float off_z = 0.0f
) {
    check_geometry_inputs(proj, angles, sid, sod, det_u0_arr, det_v0_arr,
                          src_x_arr, src_y_arr, src_z_arr);
    TORCH_CHECK(grad_vol.is_cuda() && grad_vol.scalar_type() == torch::kFloat32 &&
                grad_vol.device() == proj.device() && grad_vol.dim() == 3,
                "grad_vol must be CUDA float32 on proj.device with shape [D,H,W]");
    check_volume_shape(grad_vol.size(0), grad_vol.size(1), grad_vol.size(2));
    TORCH_CHECK(std::isfinite(pixel_size) && pixel_size > 0.0f &&
                std::isfinite(voxel_size) && voxel_size > 0.0f &&
                std::isfinite(off_x) && std::isfinite(off_y) && std::isfinite(off_z),
                "sizes must be finite and positive; ROI offsets must be finite");
    const c10::cuda::CUDAGuard device_guard(proj.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(proj.get_device());
    TORCH_CHECK(grad_vol.is_cuda() && proj.is_cuda() && angles.is_cuda() &&
                sid.is_cuda() && sod.is_cuda() && det_u0_arr.is_cuda() &&
                det_v0_arr.is_cuda() && src_x_arr.is_cuda() &&
                src_y_arr.is_cuda() && src_z_arr.is_cuda(),
                "All inputs must be CUDA tensors");
    TORCH_CHECK(grad_vol.scalar_type() == torch::kFloat32 &&
                proj.scalar_type() == torch::kFloat32 &&
                angles.scalar_type() == torch::kFloat32 &&
                sid.scalar_type() == torch::kFloat32 && sod.scalar_type() == torch::kFloat32 &&
                det_u0_arr.scalar_type() == torch::kFloat32 &&
                det_v0_arr.scalar_type() == torch::kFloat32 &&
                src_x_arr.scalar_type() == torch::kFloat32 &&
                src_y_arr.scalar_type() == torch::kFloat32 &&
                src_z_arr.scalar_type() == torch::kFloat32,
                "All inputs must be float32");
    TORCH_CHECK(pixel_size > 0.0f && voxel_size > 0.0f,
                "pixel_size and voxel_size must be positive");

    auto grad_vol_c = grad_vol.contiguous();
    auto proj_c = proj.contiguous();
    auto angles_c = angles.contiguous();
    auto sid_c = sid.contiguous();
    auto sod_c = sod.contiguous();
    auto det_u0_c = det_u0_arr.contiguous();
    auto det_v0_c = det_v0_arr.contiguous();
    auto src_x_c = src_x_arr.contiguous();
    auto src_y_c = src_y_arr.contiguous();
    auto src_z_c = src_z_arr.contiguous();

    const int A = (int)proj_c.size(0);
    const int U_det = (int)proj_c.size(1);
    const int V_det = (int)proj_c.size(2);
    const int D = (int)grad_vol_c.size(0);
    const int H = (int)grad_vol_c.size(1);
    const int W = (int)grad_vol_c.size(2);
    TORCH_CHECK(angles_c.numel() == A && sid_c.numel() == A && sod_c.numel() == A &&
                det_u0_c.numel() == A && det_v0_c.numel() == A &&
                src_x_c.numel() == A && src_y_c.numel() == A && src_z_c.numel() == A,
                "Every geometry array must contain A elements");

    auto grad_proj = torch::zeros_like(proj_c);
    auto grad_angles = torch::zeros_like(angles_c);
    auto grad_sid = torch::zeros_like(sid_c);
    auto grad_sod = torch::zeros_like(sod_c);
    auto grad_det_u0 = torch::zeros_like(det_u0_c);
    auto grad_det_v0 = torch::zeros_like(det_v0_c);
    auto grad_src_x = torch::zeros_like(src_x_c);
    auto grad_src_y = torch::zeros_like(src_y_c);
    auto grad_src_z = torch::zeros_like(src_z_c);

    const dim3 threads(8, 8, 8);
    const dim3 blocks((W + threads.x - 1) / threads.x,
                      (H + threads.y - 1) / threads.y,
                      (D + threads.z - 1) / threads.z);

    backproject_backward_kernel_cone_beam_source_only<<<blocks, threads, 0, stream>>>(
        grad_vol_c.data_ptr<float>(), proj_c.data_ptr<float>(),
        angles_c.data_ptr<float>(), sid_c.data_ptr<float>(), sod_c.data_ptr<float>(),
        det_u0_c.data_ptr<float>(), det_v0_c.data_ptr<float>(),
        src_x_c.data_ptr<float>(), src_y_c.data_ptr<float>(), src_z_c.data_ptr<float>(),
        pixel_size,
        grad_proj.data_ptr<float>(), grad_angles.data_ptr<float>(),
        grad_sid.data_ptr<float>(), grad_sod.data_ptr<float>(),
        grad_det_u0.data_ptr<float>(), grad_det_v0.data_ptr<float>(),
        grad_src_x.data_ptr<float>(), grad_src_y.data_ptr<float>(), grad_src_z.data_ptr<float>(),
        A, U_det, V_det, D, H, W, voxel_size, off_x, off_y, off_z);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_proj, grad_angles, grad_sid, grad_sod,
            grad_det_u0, grad_det_v0, grad_src_x, grad_src_y, grad_src_z};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda,
          "Cone-beam backprojection forward with source-only offsets",
          py::arg("proj"), py::arg("angles"), py::arg("sid"), py::arg("sod"),
          py::arg("det_u0_arr"), py::arg("det_v0_arr"),
          py::arg("src_x_arr"), py::arg("src_y_arr"), py::arg("src_z_arr"),
          py::arg("pixel_size"), py::arg("D"), py::arg("H"), py::arg("W"),
          py::arg("voxel_size"), py::arg("off_x") = 0.0f,
          py::arg("off_y") = 0.0f, py::arg("off_z") = 0.0f);

    m.def("backward", &backward_cuda,
          "Cone-beam backprojection backward with source-only offsets (9 gradients)",
          py::arg("grad_vol"), py::arg("proj"), py::arg("angles"),
          py::arg("sid"), py::arg("sod"),
          py::arg("det_u0_arr"), py::arg("det_v0_arr"),
          py::arg("src_x_arr"), py::arg("src_y_arr"), py::arg("src_z_arr"),
          py::arg("pixel_size"), py::arg("voxel_size"),
          py::arg("off_x") = 0.0f, py::arg("off_y") = 0.0f,
          py::arg("off_z") = 0.0f);
}
