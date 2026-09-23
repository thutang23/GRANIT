#ifndef CUDADEM_RIGID_POSE_TRAJECTORY_H
#define CUDADEM_RIGID_POSE_TRAJECTORY_H

// Strict host-side reader for the canonical RIGIDPS2 wall trajectory.
// The pose maps a body-coordinate point b into world coordinates as
//   x_world = translation + R_body_to_world * (b - reference_origin).
// This keeps the reference origin explicit instead of relying on an implicit
// OBJ or YAML convention.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudadem {

struct RigidPoseSample {
    std::array<double, 3> translation{};
    std::array<double, 4> quaternion_wxyz{1.0, 0.0, 0.0, 0.0};
    std::array<double, 3> linear_velocity{};
    std::array<double, 3> angular_velocity{}; // world frame
};

class RigidPoseTrajectory {
public:
    bool load(const char* path) {
        if (!path || !*path) throw std::runtime_error("RIGIDPS2 path is empty");
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error(std::string("cannot open RIGIDPS2: ") + path);
        char magic[8]; uint32_t version = 0, count = 0, policy = 0, reserved = 0;
        file.read(magic, 8); file.read(reinterpret_cast<char*>(&version), 4);
        file.read(reinterpret_cast<char*>(&count), 4);
        file.read(reinterpret_cast<char*>(reference_origin_.data()), 3 * sizeof(double));
        file.read(reinterpret_cast<char*>(&policy), 4); file.read(reinterpret_cast<char*>(&reserved), 4);
        if (!file || std::memcmp(magic, "RIGIDPS2", 8) != 0 || version != 2 ||
            count == 0 || count > 1000000 || policy > 1 || reserved != 0 || !finite(reference_origin_)) {
            throw std::runtime_error("invalid RIGIDPS2 header");
        }
        file.seekg(0, std::ios::end);
        constexpr std::streamoff header_bytes = 48, record_bytes = 64;
        if (file.tellg() != header_bytes + static_cast<std::streamoff>(count) * record_bytes)
            throw std::runtime_error("RIGIDPS2 byte count is not exact");
        file.seekg(header_bytes, std::ios::beg);
        times_.resize(count); translations_.resize(count); quaternions_.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            double r[8]; file.read(reinterpret_cast<char*>(r), sizeof(r));
            if (!file) throw std::runtime_error("truncated RIGIDPS2 pose");
            times_[i] = r[0]; translations_[i] = {r[1], r[2], r[3]};
            quaternions_[i] = {r[4], r[5], r[6], r[7]};
            if (!std::isfinite(times_[i]) || !finite(translations_[i]) || !finite(quaternions_[i]) ||
                (i && times_[i] <= times_[i - 1]))
                throw std::runtime_error("RIGIDPS2 requires finite, strictly increasing poses");
            normalize(quaternions_[i]);
        }
        hold_out_of_range_ = policy == 1;
        return true;
    }

    RigidPoseSample sample(double time) const {
        if (times_.empty() || !std::isfinite(time)) throw std::runtime_error("invalid RIGIDPS2 sample time");
        if (time < times_.front() || time > times_.back()) {
            if (!hold_out_of_range_) throw std::runtime_error("RIGIDPS2 does not cover simulation time");
            const size_t i = time < times_.front() ? 0 : times_.size() - 1;
            return {translations_[i], quaternions_[i], {}, {}};
        }
        auto upper = std::upper_bound(times_.begin(), times_.end(), time);
        const size_t hi = static_cast<size_t>(upper - times_.begin());
        if (hi == 0 || hi == times_.size()) {
            const size_t i = hi == 0 ? 0 : times_.size() - 1;
            return {translations_[i], quaternions_[i], {}, {}};
        }
        const size_t lo = hi - 1;
        const double dt = times_[hi] - times_[lo], a = (time - times_[lo]) / dt;
        RigidPoseSample result;
        for (int k = 0; k < 3; ++k) {
            result.translation[k] = translations_[lo][k] + a * (translations_[hi][k] - translations_[lo][k]);
            result.linear_velocity[k] = (translations_[hi][k] - translations_[lo][k]) / dt;
        }
        result.quaternion_wxyz = slerp(quaternions_[lo], quaternions_[hi], a);
        result.angular_velocity = angular_velocity(quaternions_[lo], quaternions_[hi], dt);
        return result;
    }

    const std::array<double, 3>& reference_origin() const { return reference_origin_; }

    static std::array<double, 9> rotation_matrix(const std::array<double, 4>& q) {
        const double w=q[0], x=q[1], y=q[2], z=q[3];
        return {1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)};
    }

private:
    static bool finite(const std::array<double, 3>& a) { return std::isfinite(a[0]) && std::isfinite(a[1]) && std::isfinite(a[2]); }
    static bool finite(const std::array<double, 4>& a) { return std::isfinite(a[0]) && std::isfinite(a[1]) && std::isfinite(a[2]) && std::isfinite(a[3]); }
    static void normalize(std::array<double, 4>& q) {
        double n=0; for (double v:q) n += v*v; n=std::sqrt(n);
        if (!(n > 1e-12)) throw std::runtime_error("RIGIDPS2 quaternion is zero");
        for (double& v:q) v/=n;
    }
    static std::array<double, 4> slerp(std::array<double, 4> a, std::array<double, 4> b, double t) {
        double dot=0; for(int i=0;i<4;++i) dot += a[i]*b[i];
        if (dot < 0) { for(double& v:b) v=-v; dot=-dot; }
        dot=std::max(-1.0,std::min(1.0,dot)); std::array<double,4> out{};
        if (dot > 0.9995) for(int i=0;i<4;++i) out[i]=a[i]+t*(b[i]-a[i]);
        else { const double theta=std::acos(dot), s=std::sin(theta); for(int i=0;i<4;++i) out[i]=(std::sin((1-t)*theta)*a[i]+std::sin(t*theta)*b[i])/s; }
        normalize(out); return out;
    }
    static std::array<double, 3> angular_velocity(std::array<double,4> a, std::array<double,4> b, double dt) {
        double dot=0; for(int i=0;i<4;++i) dot+=a[i]*b[i]; if(dot<0) for(double& v:b)v=-v;
        // b * conjugate(a): shortest world-frame body-to-world increment.
        const double rw=b[0]*a[0]+b[1]*a[1]+b[2]*a[2]+b[3]*a[3];
        const std::array<double,3> r={-b[0]*a[1]+b[1]*a[0]-b[2]*a[3]+b[3]*a[2],
            -b[0]*a[2]+b[2]*a[0]-b[3]*a[1]+b[1]*a[3], -b[0]*a[3]+b[3]*a[0]-b[1]*a[2]+b[2]*a[1]};
        const double s=std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
        if(s < 1e-14) return {};
        const double scale=2*std::atan2(s,std::max(rw,0.0))/(s*dt);
        return {r[0]*scale,r[1]*scale,r[2]*scale};
    }
    std::vector<double> times_;
    std::vector<std::array<double,3>> translations_;
    std::vector<std::array<double,4>> quaternions_;
    std::array<double,3> reference_origin_{};
    bool hold_out_of_range_ = false;
};

} // namespace cudadem
#endif
