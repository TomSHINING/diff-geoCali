#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <vector>

namespace py = pybind11;

/*
Source-only cone-beam forward projector
=======================================

This file uses exactly the same geometry convention as
cone_beam_source_only.cu (the source-only backprojector):

  1. Projection layout is [A, U, V].
  2. beta rotates world coordinates to detector-local coordinates with

         px =  x*cos(beta) + y*sin(beta)
         py = -x*sin(beta) + y*cos(beta).

  3. The nominal source and fixed detector plane in local coordinates are

         source   = (0, sod,       0)
         detector = (0, sod - sid, 0).

  4. src_x/src_y/src_z are WORLD-coordinate source offsets.  They move only
     the source; the detector centre and plane do not move with them.
  5. det_u0/det_v0 are detector centre indices.  Detector pixel (u,v) has

         u_phys = (u - det_u0) * pixel_size
         v_phys = (v - det_v0) * pixel_size.

The resulting world-coordinate ray is

  O = (-sod*sin(beta) + src_x,
        sod*cos(beta) + src_y,
        src_z)

  T = (u_phys*cos(beta) - (sod-sid)*sin(beta),
       u_phys*sin(beta) + (sod-sid)*cos(beta),
       v_phys)

  D = normalize(T - O).

The forward value is a physical line integral through the volume.  The paired
backprojector is FDK-style and contains an additional geometric weight, so the
two operators share geometry but are intentionally not a strict adjoint pair.
*/


__device__ __forceinline__ bool update_slab(
    float origin,
    float direction,
    float slab_min,
    float slab_max,
    int axis,
    float& t_min,
    float& t_max,
    int& entry_axis
) {
    if (fabsf(direction) < 1e-12f) {
        return origin >= slab_min && origin <= slab_max;
    }

    const float inv_direction = 1.0f / direction;
    float t1 = (slab_min - origin) * inv_direction;
    float t2 = (slab_max - origin) * inv_direction;
    if (t1 > t2) {
        const float temp = t1;
        t1 = t2;
        t2 = temp;
    }
    if (t1 > t_min) {
        t_min = t1;
        entry_axis = axis;
    }
    t_max = fminf(t_max, t2);
    return t_max >= t_min;
}


__device__ __forceinline__ bool intersect_aabb(
    const float3 origin,
    const float3 direction,
    const float3 box_min,
    const float3 box_max,
    float& t_min,
    float& t_max,
    int& entry_axis
) {
    t_min = 0.0f;
    t_max = FLT_MAX;
    entry_axis = -1;

    if (!update_slab(origin.x, direction.x, box_min.x, box_max.x,
                     0, t_min, t_max, entry_axis)) {
        return false;
    }
    if (!update_slab(origin.y, direction.y, box_min.y, box_max.y,
                     1, t_min, t_max, entry_axis)) {
        return false;
    }
    if (!update_slab(origin.z, direction.z, box_min.z, box_max.z,
                     2, t_min, t_max, entry_axis)) {
        return false;
    }
    return t_max >= t_min;
}


__device__ __forceinline__ bool make_source_only_ray(
    int angle_index,
    int detector_u,
    int detector_v,
    const float* __restrict__ angles,
    const float* __restrict__ sid,
    const float* __restrict__ sod,
    const float* __restrict__ det_u0,
    const float* __restrict__ det_v0,
    const float* __restrict__ src_x,
    const float* __restrict__ src_y,
    const float* __restrict__ src_z,
    float pixel_size,
    float3& origin,
    float3& target,
    float3& direction,
    float& ray_length
) {
    const float beta = angles[angle_index];
    const float c = cosf(beta);
    const float s = sinf(beta);
    const float sod_a = sod[angle_index];
    const float detector_n = sod_a - sid[angle_index];
    const float u_phys = (detector_u - det_u0[angle_index]) * pixel_size;
    const float v_phys = (detector_v - det_v0[angle_index]) * pixel_size;

    origin = make_float3(
        -sod_a * s + src_x[angle_index],
         sod_a * c + src_y[angle_index],
         src_z[angle_index]
    );

    target = make_float3(
        u_phys * c - detector_n * s,
        u_phys * s + detector_n * c,
        v_phys
    );

    const float3 ray = make_float3(
        target.x - origin.x,
        target.y - origin.y,
        target.z - origin.z
    );
    ray_length = sqrtf(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
    if (ray_length <= 1e-8f) {
        return false;
    }

    const float inv_length = 1.0f / ray_length;
    direction = make_float3(
        ray.x * inv_length,
        ray.y * inv_length,
        ray.z * inv_length
    );
    return true;
}


__device__ __forceinline__ float sample_volume_trilinear_with_grad(
    const float* __restrict__ volume,
    float* __restrict__ grad_volume,
    int W,
    int H,
    int D,
    float x,
    float y,
    float z,
    float3& grad_spatial,
    float grad_sample,
    bool compute_grad_volume
) {
    int x0_raw = __float2int_rd(x);
    int y0_raw = __float2int_rd(y);
    int z0_raw = __float2int_rd(z);
    const float fx = x - x0_raw;
    const float fy = y - y0_raw;
    const float fz = z - z0_raw;

    int x0 = max(0, min(x0_raw, W - 1));
    int y0 = max(0, min(y0_raw, H - 1));
    int z0 = max(0, min(z0_raw, D - 1));
    int x1 = max(0, min(x0_raw + 1, W - 1));
    int y1 = max(0, min(y0_raw + 1, H - 1));
    int z1 = max(0, min(z0_raw + 1, D - 1));

    const int64_t plane = static_cast<int64_t>(H) * W;
    const int64_t i000 = static_cast<int64_t>(z0) * plane + static_cast<int64_t>(y0) * W + x0;
    const int64_t i100 = static_cast<int64_t>(z0) * plane + static_cast<int64_t>(y0) * W + x1;
    const int64_t i010 = static_cast<int64_t>(z0) * plane + static_cast<int64_t>(y1) * W + x0;
    const int64_t i110 = static_cast<int64_t>(z0) * plane + static_cast<int64_t>(y1) * W + x1;
    const int64_t i001 = static_cast<int64_t>(z1) * plane + static_cast<int64_t>(y0) * W + x0;
    const int64_t i101 = static_cast<int64_t>(z1) * plane + static_cast<int64_t>(y0) * W + x1;
    const int64_t i011 = static_cast<int64_t>(z1) * plane + static_cast<int64_t>(y1) * W + x0;
    const int64_t i111 = static_cast<int64_t>(z1) * plane + static_cast<int64_t>(y1) * W + x1;

    const float v000 = volume[i000];
    const float v100 = volume[i100];
    const float v010 = volume[i010];
    const float v110 = volume[i110];
    const float v001 = volume[i001];
    const float v101 = volume[i101];
    const float v011 = volume[i011];
    const float v111 = volume[i111];

    const float one_minus_fx = 1.0f - fx;
    const float one_minus_fy = 1.0f - fy;
    const float one_minus_fz = 1.0f - fz;
    const float w000 = one_minus_fx * one_minus_fy * one_minus_fz;
    const float w100 = fx * one_minus_fy * one_minus_fz;
    const float w010 = one_minus_fx * fy * one_minus_fz;
    const float w110 = fx * fy * one_minus_fz;
    const float w001 = one_minus_fx * one_minus_fy * fz;
    const float w101 = fx * one_minus_fy * fz;
    const float w011 = one_minus_fx * fy * fz;
    const float w111 = fx * fy * fz;

    if (compute_grad_volume && grad_volume != nullptr) {
        atomicAdd(&grad_volume[i000], grad_sample * w000);
        atomicAdd(&grad_volume[i100], grad_sample * w100);
        atomicAdd(&grad_volume[i010], grad_sample * w010);
        atomicAdd(&grad_volume[i110], grad_sample * w110);
        atomicAdd(&grad_volume[i001], grad_sample * w001);
        atomicAdd(&grad_volume[i101], grad_sample * w101);
        atomicAdd(&grad_volume[i011], grad_sample * w011);
        atomicAdd(&grad_volume[i111], grad_sample * w111);
    }

    grad_spatial.x =
        one_minus_fz * (one_minus_fy * (v100 - v000) + fy * (v110 - v010)) +
        fz * (one_minus_fy * (v101 - v001) + fy * (v111 - v011));
    grad_spatial.y =
        one_minus_fz * (one_minus_fx * (v010 - v000) + fx * (v110 - v100)) +
        fz * (one_minus_fx * (v011 - v001) + fx * (v111 - v101));
    grad_spatial.z =
        one_minus_fy * (one_minus_fx * (v001 - v000) + fx * (v101 - v100)) +
        fy * (one_minus_fx * (v011 - v010) + fx * (v111 - v110));

    return
        v000 * w000 + v100 * w100 + v010 * w010 + v110 * w110 +
        v001 * w001 + v101 * w101 + v011 * w011 + v111 * w111;
}


__global__ void source_only_forward_kernel(
    const float* __restrict__ volume,
    const float* __restrict__ angles,
    const float* __restrict__ sid,
    const float* __restrict__ sod,
    const float* __restrict__ det_u0,
    const float* __restrict__ det_v0,
    const float* __restrict__ src_x,
    const float* __restrict__ src_y,
    const float* __restrict__ src_z,
    float pixel_size,
    float* __restrict__ projections,
    int D,
    int H,
    int W,
    int A,
    int U,
    int V,
    float step_size,
    float voxel_size,
    float off_x,
    float off_y,
    float off_z
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_rays = static_cast<int64_t>(A) * U * V;
    if (index >= total_rays) return;

    const int detector_v = static_cast<int>(index % V);
    const int64_t index_without_v = index / V;
    const int detector_u = static_cast<int>(index_without_v % U);
    const int angle_index = static_cast<int>(index_without_v / U);

    float3 origin, target, direction;
    float ray_length;
    if (!make_source_only_ray(
            angle_index, detector_u, detector_v,
            angles, sid, sod, det_u0, det_v0, src_x, src_y, src_z,
            pixel_size, origin, target, direction, ray_length)) {
        projections[index] = 0.0f;
        return;
    }

    // Volume samples are centred at (i - size/2) * voxel_size + offset.
    // The physical support extends half a voxel beyond the first/last centre.
    const float3 box_min = make_float3(
        (-W * 0.5f - 0.5f) * voxel_size + off_x,
        (-H * 0.5f - 0.5f) * voxel_size + off_y,
        (-D * 0.5f - 0.5f) * voxel_size + off_z
    );
    const float3 box_max = make_float3(
        ( W * 0.5f - 0.5f) * voxel_size + off_x,
        ( H * 0.5f - 0.5f) * voxel_size + off_y,
        ( D * 0.5f - 0.5f) * voxel_size + off_z
    );

    float t_min, t_max;
    int entry_axis;
    float integral = 0.0f;
    if (intersect_aabb(
            origin, direction, box_min, box_max,
            t_min, t_max, entry_axis)) {
        // Keep the fixed-step rule used by the original differentiable ray caster.
        for (float t = t_min; t <= t_max; t += step_size) {
            const float px = origin.x + t * direction.x;
            const float py = origin.y + t * direction.y;
            const float pz = origin.z + t * direction.z;

            const float voxel_x = (px - off_x) / voxel_size + W * 0.5f;
            const float voxel_y = (py - off_y) / voxel_size + H * 0.5f;
            const float voxel_z = (pz - off_z) / voxel_size + D * 0.5f;
            float3 unused_gradient;
            integral += sample_volume_trilinear_with_grad(
                volume, nullptr, W, H, D,
                voxel_x, voxel_y, voxel_z,
                unused_gradient, 0.0f, false
            );
        }
        integral *= step_size;
    }
    projections[index] = integral;
}


__global__ void source_only_backward_kernel(
    const float* __restrict__ grad_projection,
    const float* __restrict__ volume,
    const float* __restrict__ angles,
    const float* __restrict__ sid,
    const float* __restrict__ sod,
    const float* __restrict__ det_u0,
    const float* __restrict__ det_v0,
    const float* __restrict__ src_x,
    const float* __restrict__ src_y,
    const float* __restrict__ src_z,
    float pixel_size,
    float* __restrict__ grad_volume,
    float* __restrict__ grad_angles,
    float* __restrict__ grad_sid,
    float* __restrict__ grad_sod,
    float* __restrict__ grad_det_u0,
    float* __restrict__ grad_det_v0,
    float* __restrict__ grad_src_x,
    float* __restrict__ grad_src_y,
    float* __restrict__ grad_src_z,
    int D,
    int H,
    int W,
    int A,
    int U,
    int V,
    float step_size,
    float voxel_size,
    bool compute_grad_volume,
    float off_x,
    float off_y,
    float off_z
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_rays = static_cast<int64_t>(A) * U * V;
    if (index >= total_rays) return;

    const float grad_output = grad_projection[index];
    if (grad_output == 0.0f) return;

    const int detector_v = static_cast<int>(index % V);
    const int64_t index_without_v = index / V;
    const int detector_u = static_cast<int>(index_without_v % U);
    const int angle_index = static_cast<int>(index_without_v / U);

    float3 origin, target, direction;
    float ray_length;
    if (!make_source_only_ray(
            angle_index, detector_u, detector_v,
            angles, sid, sod, det_u0, det_v0, src_x, src_y, src_z,
            pixel_size, origin, target, direction, ray_length)) {
        return;
    }

    const float3 box_min = make_float3(
        (-W * 0.5f - 0.5f) * voxel_size + off_x,
        (-H * 0.5f - 0.5f) * voxel_size + off_y,
        (-D * 0.5f - 0.5f) * voxel_size + off_z
    );
    const float3 box_max = make_float3(
        ( W * 0.5f - 0.5f) * voxel_size + off_x,
        ( H * 0.5f - 0.5f) * voxel_size + off_y,
        ( D * 0.5f - 0.5f) * voxel_size + off_z
    );

    float t_min, t_max;
    int entry_axis;
    float3 grad_origin_direct = make_float3(0.0f, 0.0f, 0.0f);
    float3 grad_direction = make_float3(0.0f, 0.0f, 0.0f);

    if (intersect_aabb(
            origin, direction, box_min, box_max,
            t_min, t_max, entry_axis)) {
        const float grad_sample = grad_output * step_size;
        const float inv_voxel_size = 1.0f / voxel_size;
        float grad_t_min = 0.0f;

        for (float t = t_min; t <= t_max; t += step_size) {
            const float px = origin.x + t * direction.x;
            const float py = origin.y + t * direction.y;
            const float pz = origin.z + t * direction.z;

            const float voxel_x = (px - off_x) * inv_voxel_size + W * 0.5f;
            const float voxel_y = (py - off_y) * inv_voxel_size + H * 0.5f;
            const float voxel_z = (pz - off_z) * inv_voxel_size + D * 0.5f;

            float3 grad_spatial_voxel;
            sample_volume_trilinear_with_grad(
                volume, grad_volume, W, H, D,
                voxel_x, voxel_y, voxel_z,
                grad_spatial_voxel, grad_sample, compute_grad_volume
            );

            const float3 grad_spatial_world = make_float3(
                grad_spatial_voxel.x * inv_voxel_size,
                grad_spatial_voxel.y * inv_voxel_size,
                grad_spatial_voxel.z * inv_voxel_size
            );
            grad_origin_direct.x += grad_sample * grad_spatial_world.x;
            grad_origin_direct.y += grad_sample * grad_spatial_world.y;
            grad_origin_direct.z += grad_sample * grad_spatial_world.z;
            grad_direction.x += grad_sample * grad_spatial_world.x * t;
            grad_direction.y += grad_sample * grad_spatial_world.y * t;
            grad_direction.z += grad_sample * grad_spatial_world.z * t;
            grad_t_min += grad_sample * (
                grad_spatial_world.x * direction.x +
                grad_spatial_world.y * direction.y +
                grad_spatial_world.z * direction.z
            );
        }

        // All sample positions are anchored at the AABB entry point.  Within
        // one slab-selection region, t_min=(bound-origin[axis])/direction[axis].
        // The exit point only changes the discrete sample count and therefore
        // has no derivative between count-change events.
        if (entry_axis == 0) {
            grad_origin_direct.x -= grad_t_min / direction.x;
            grad_direction.x -= grad_t_min * t_min / direction.x;
        } else if (entry_axis == 1) {
            grad_origin_direct.y -= grad_t_min / direction.y;
            grad_direction.y -= grad_t_min * t_min / direction.y;
        } else if (entry_axis == 2) {
            grad_origin_direct.z -= grad_t_min / direction.z;
            grad_direction.z -= grad_t_min * t_min / direction.z;
        }
    }

    // Chain through direction = normalize(target - origin).
    const float direction_dot_grad =
        direction.x * grad_direction.x +
        direction.y * grad_direction.y +
        direction.z * grad_direction.z;
    const float inv_ray_length = 1.0f / ray_length;
    const float3 grad_ray = make_float3(
        (grad_direction.x - direction.x * direction_dot_grad) * inv_ray_length,
        (grad_direction.y - direction.y * direction_dot_grad) * inv_ray_length,
        (grad_direction.z - direction.z * direction_dot_grad) * inv_ray_length
    );
    const float3 grad_origin = make_float3(
        grad_origin_direct.x - grad_ray.x,
        grad_origin_direct.y - grad_ray.y,
        grad_origin_direct.z - grad_ray.z
    );
    const float3 grad_target = grad_ray;

    const float beta = angles[angle_index];
    const float c = cosf(beta);
    const float s = sinf(beta);
    const float sod_a = sod[angle_index];
    const float detector_n = sod_a - sid[angle_index];
    const float u_phys = (detector_u - det_u0[angle_index]) * pixel_size;

    // angle: world-coordinate source offsets are fixed and do not rotate.
    const float dO_x_dbeta = -sod_a * c;
    const float dO_y_dbeta = -sod_a * s;
    const float dT_x_dbeta = -u_phys * s - detector_n * c;
    const float dT_y_dbeta =  u_phys * c - detector_n * s;
    const float grad_beta =
        grad_origin.x * dO_x_dbeta + grad_origin.y * dO_y_dbeta +
        grad_target.x * dT_x_dbeta + grad_target.y * dT_y_dbeta;
    atomicAdd(&grad_angles[angle_index], grad_beta);

    // sid moves only the fixed detector plane; the source remains fixed.
    const float grad_sid_value = grad_target.x * s - grad_target.y * c;
    atomicAdd(&grad_sid[angle_index], grad_sid_value);

    // sod translates the nominal source and detector together in the radial direction.
    const float radial_grad_x = -s;
    const float radial_grad_y = c;
    const float grad_sod_value =
        (grad_origin.x + grad_target.x) * radial_grad_x +
        (grad_origin.y + grad_target.y) * radial_grad_y;
    atomicAdd(&grad_sod[angle_index], grad_sod_value);

    // Detector centre inputs are measured in pixel-index units.
    const float grad_det_u0_value =
        -pixel_size * (grad_target.x * c + grad_target.y * s);
    const float grad_det_v0_value = -pixel_size * grad_target.z;
    atomicAdd(&grad_det_u0[angle_index], grad_det_u0_value);
    atomicAdd(&grad_det_v0[angle_index], grad_det_v0_value);

    // World-coordinate source-only offsets.
    atomicAdd(&grad_src_x[angle_index], grad_origin.x);
    atomicAdd(&grad_src_y[angle_index], grad_origin.y);
    atomicAdd(&grad_src_z[angle_index], grad_origin.z);
}


static void check_geometry_inputs(
    const torch::Tensor& reference,
    const torch::Tensor& angles,
    const torch::Tensor& sid,
    const torch::Tensor& sod,
    const torch::Tensor& det_u0,
    const torch::Tensor& det_v0,
    const torch::Tensor& src_x,
    const torch::Tensor& src_y,
    const torch::Tensor& src_z,
    int64_t expected_angles
) {
    TORCH_CHECK(reference.is_cuda(), "reference tensor must be CUDA");
    TORCH_CHECK(reference.scalar_type() == torch::kFloat32,
                "reference tensor must be float32");

    const torch::Tensor tensors[] = {
        angles, sid, sod, det_u0, det_v0, src_x, src_y, src_z
    };
    const char* names[] = {
        "angles", "sid", "sod", "det_u0", "det_v0",
        "src_x", "src_y", "src_z"
    };
    for (int i = 0; i < 8; ++i) {
        TORCH_CHECK(tensors[i].is_cuda(), names[i], " must be CUDA");
        TORCH_CHECK(tensors[i].scalar_type() == torch::kFloat32,
                    names[i], " must be float32");
        TORCH_CHECK(tensors[i].device() == reference.device(),
                    names[i], " must be on the same CUDA device as the reference tensor");
        TORCH_CHECK(tensors[i].numel() == expected_angles,
                    names[i], " must contain A elements");
    }
}


torch::Tensor forward_cuda(
    torch::Tensor volume,
    torch::Tensor angles,
    torch::Tensor sid,
    torch::Tensor sod,
    torch::Tensor det_u0,
    torch::Tensor det_v0,
    torch::Tensor src_x,
    torch::Tensor src_y,
    torch::Tensor src_z,
    float pixel_size,
    int U,
    int V,
    float step_size,
    float voxel_size,
    float off_x = 0.0f,
    float off_y = 0.0f,
    float off_z = 0.0f
) {
    TORCH_CHECK(volume.dim() == 3, "volume must have shape [D,H,W]");
    TORCH_CHECK(U > 0 && V > 0, "U and V must be positive");
    TORCH_CHECK(pixel_size > 0.0f && step_size > 0.0f && voxel_size > 0.0f,
                "pixel_size, step_size and voxel_size must be positive");

    const int A = static_cast<int>(angles.numel());
    TORCH_CHECK(A > 0, "angles must contain at least one view");
    check_geometry_inputs(
        volume, angles, sid, sod, det_u0, det_v0, src_x, src_y, src_z, A
    );

    auto volume_c = volume.contiguous();
    auto angles_c = angles.contiguous();
    auto sid_c = sid.contiguous();
    auto sod_c = sod.contiguous();
    auto det_u0_c = det_u0.contiguous();
    auto det_v0_c = det_v0.contiguous();
    auto src_x_c = src_x.contiguous();
    auto src_y_c = src_y.contiguous();
    auto src_z_c = src_z.contiguous();

    const int D = static_cast<int>(volume_c.size(0));
    const int H = static_cast<int>(volume_c.size(1));
    const int W = static_cast<int>(volume_c.size(2));
    TORCH_CHECK(D > 0 && H > 0 && W > 0,
                "every volume dimension must be positive");
    auto projections = torch::zeros({A, U, V}, volume_c.options());

    const int threads = 256;
    const int64_t total_rays = static_cast<int64_t>(A) * U * V;
    const int blocks = static_cast<int>((total_rays + threads - 1) / threads);
    const c10::cuda::CUDAGuard device_guard(volume_c.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    source_only_forward_kernel<<<blocks, threads, 0, stream>>>(
        volume_c.data_ptr<float>(), angles_c.data_ptr<float>(),
        sid_c.data_ptr<float>(), sod_c.data_ptr<float>(),
        det_u0_c.data_ptr<float>(), det_v0_c.data_ptr<float>(),
        src_x_c.data_ptr<float>(), src_y_c.data_ptr<float>(), src_z_c.data_ptr<float>(),
        pixel_size, projections.data_ptr<float>(),
        D, H, W, A, U, V, step_size, voxel_size, off_x, off_y, off_z
    );
    const cudaError_t launch_error = cudaGetLastError();
    TORCH_CHECK(launch_error == cudaSuccess,
                "source-only forward kernel launch failed: ",
                cudaGetErrorString(launch_error));
    return projections;
}


std::vector<torch::Tensor> backward_cuda(
    torch::Tensor grad_projection,
    torch::Tensor volume,
    torch::Tensor angles,
    torch::Tensor sid,
    torch::Tensor sod,
    torch::Tensor det_u0,
    torch::Tensor det_v0,
    torch::Tensor src_x,
    torch::Tensor src_y,
    torch::Tensor src_z,
    float pixel_size,
    float step_size,
    float voxel_size,
    bool compute_grad_volume = true,
    float off_x = 0.0f,
    float off_y = 0.0f,
    float off_z = 0.0f
) {
    TORCH_CHECK(volume.dim() == 3, "volume must have shape [D,H,W]");
    TORCH_CHECK(grad_projection.dim() == 3,
                "grad_projection must have shape [A,U,V]");
    TORCH_CHECK(grad_projection.is_cuda() &&
                grad_projection.scalar_type() == torch::kFloat32,
                "grad_projection must be a float32 CUDA tensor");
    TORCH_CHECK(grad_projection.device() == volume.device(),
                "grad_projection and volume must be on the same CUDA device");
    TORCH_CHECK(pixel_size > 0.0f && step_size > 0.0f && voxel_size > 0.0f,
                "pixel_size, step_size and voxel_size must be positive");

    const int A = static_cast<int>(grad_projection.size(0));
    const int U = static_cast<int>(grad_projection.size(1));
    const int V = static_cast<int>(grad_projection.size(2));
    TORCH_CHECK(A > 0 && U > 0 && V > 0,
                "every grad_projection dimension must be positive");
    check_geometry_inputs(
        volume, angles, sid, sod, det_u0, det_v0, src_x, src_y, src_z, A
    );

    auto grad_projection_c = grad_projection.contiguous();
    auto volume_c = volume.contiguous();
    auto angles_c = angles.contiguous();
    auto sid_c = sid.contiguous();
    auto sod_c = sod.contiguous();
    auto det_u0_c = det_u0.contiguous();
    auto det_v0_c = det_v0.contiguous();
    auto src_x_c = src_x.contiguous();
    auto src_y_c = src_y.contiguous();
    auto src_z_c = src_z.contiguous();

    const int D = static_cast<int>(volume_c.size(0));
    const int H = static_cast<int>(volume_c.size(1));
    const int W = static_cast<int>(volume_c.size(2));
    TORCH_CHECK(D > 0 && H > 0 && W > 0,
                "every volume dimension must be positive");

    auto grad_volume = compute_grad_volume
        ? torch::zeros_like(volume_c)
        : torch::empty({0}, volume_c.options());
    auto grad_angles = torch::zeros_like(angles_c);
    auto grad_sid = torch::zeros_like(sid_c);
    auto grad_sod = torch::zeros_like(sod_c);
    auto grad_det_u0 = torch::zeros_like(det_u0_c);
    auto grad_det_v0 = torch::zeros_like(det_v0_c);
    auto grad_src_x = torch::zeros_like(src_x_c);
    auto grad_src_y = torch::zeros_like(src_y_c);
    auto grad_src_z = torch::zeros_like(src_z_c);

    const int threads = 256;
    const int64_t total_rays = static_cast<int64_t>(A) * U * V;
    const int blocks = static_cast<int>((total_rays + threads - 1) / threads);
    const c10::cuda::CUDAGuard device_guard(volume_c.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    source_only_backward_kernel<<<blocks, threads, 0, stream>>>(
        grad_projection_c.data_ptr<float>(), volume_c.data_ptr<float>(),
        angles_c.data_ptr<float>(), sid_c.data_ptr<float>(), sod_c.data_ptr<float>(),
        det_u0_c.data_ptr<float>(), det_v0_c.data_ptr<float>(),
        src_x_c.data_ptr<float>(), src_y_c.data_ptr<float>(), src_z_c.data_ptr<float>(),
        pixel_size,
        compute_grad_volume ? grad_volume.data_ptr<float>() : nullptr,
        grad_angles.data_ptr<float>(), grad_sid.data_ptr<float>(), grad_sod.data_ptr<float>(),
        grad_det_u0.data_ptr<float>(), grad_det_v0.data_ptr<float>(),
        grad_src_x.data_ptr<float>(), grad_src_y.data_ptr<float>(), grad_src_z.data_ptr<float>(),
        D, H, W, A, U, V, step_size, voxel_size, compute_grad_volume,
        off_x, off_y, off_z
    );
    const cudaError_t launch_error = cudaGetLastError();
    TORCH_CHECK(launch_error == cudaSuccess,
                "source-only backward kernel launch failed: ",
                cudaGetErrorString(launch_error));

    return {
        grad_volume, grad_angles, grad_sid, grad_sod,
        grad_det_u0, grad_det_v0, grad_src_x, grad_src_y, grad_src_z
    };
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "forward", &forward_cuda,
        "Source-only cone-beam forward projection",
        py::arg("volume"), py::arg("angles"),
        py::arg("sid"), py::arg("sod"),
        py::arg("det_u0_arr"), py::arg("det_v0_arr"),
        py::arg("src_x_arr"), py::arg("src_y_arr"), py::arg("src_z_arr"),
        py::arg("pixel_size"), py::arg("U"), py::arg("V"),
        py::arg("step_size"), py::arg("voxel_size"),
        py::arg("off_x") = 0.0f, py::arg("off_y") = 0.0f,
        py::arg("off_z") = 0.0f
    );

    module.def(
        "backward", &backward_cuda,
        "Source-only cone-beam forward-projector backward (9 gradients)",
        py::arg("grad_projection"), py::arg("volume"), py::arg("angles"),
        py::arg("sid"), py::arg("sod"),
        py::arg("det_u0_arr"), py::arg("det_v0_arr"),
        py::arg("src_x_arr"), py::arg("src_y_arr"), py::arg("src_z_arr"),
        py::arg("pixel_size"), py::arg("step_size"), py::arg("voxel_size"),
        py::arg("compute_grad_volume") = true,
        py::arg("off_x") = 0.0f, py::arg("off_y") = 0.0f,
        py::arg("off_z") = 0.0f
    );
}
