#ifndef GRANIT_CHECKPOINT_TRANSACTION_H
#define GRANIT_CHECKPOINT_TRANSACTION_H

// POSIX checkpoint publication: immutable payloads and sidecars are flushed
// before one atomic symlink rename publishes the entire generation.  Consumers
// must resolve the .bin alias once before opening any adjacent sidecar.
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>

namespace cudadem {
namespace checkpoint_detail {
inline void sync_path(const std::filesystem::path& path, bool directory = false) {
    const int fd = ::open(path.c_str(), O_RDONLY | (directory ? O_DIRECTORY : 0));
    if (fd < 0) throw std::runtime_error("Cannot open checkpoint for fsync: " + path.string());
    const int result = ::fsync(fd);
    const int saved_errno = errno;
    const int closed = ::close(fd);
    if (result != 0 || closed != 0)
        throw std::runtime_error("Cannot sync checkpoint: " + path.string() + ": " + std::strerror(saved_errno));
}
inline uint64_t checksum(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot read checkpoint member: " + path.string());
    uint64_t hash = UINT64_C(14695981039346656037);
    char buffer[65536];
    while (in) {
        in.read(buffer, sizeof(buffer));
        for (std::streamsize i = 0; i < in.gcount(); ++i) {
            hash ^= static_cast<unsigned char>(buffer[i]);
            hash *= UINT64_C(1099511628211);
        }
    }
    if (!in.eof()) throw std::runtime_error("Failed reading checkpoint member: " + path.string());
    return hash;
}
inline bool valid_member_name(const std::string& name) {
    return name == "checkpoint.bin" ||
        (name.rfind("checkpoint.bin.", 0) == 0 && name.find('/') == std::string::npos &&
         name.find_first_of(" \t\r\n") == std::string::npos &&
         name.size() > std::strlen("checkpoint.bin."));
}
inline bool is_generation(const std::filesystem::path& path) {
    return path.parent_path().filename().string().rfind(".checkpoint-generation-", 0) == 0;
}
} // namespace checkpoint_detail

// Validate the whole bundle, including arbitrary solver-specific sidecars.
// Legacy regular .bin files are accepted; newly generated bundles require the
// manifest even when explicitly addressed by their hidden payload path.
inline std::string resolve_checkpoint_path(const std::string& requested) {
    namespace fs = std::filesystem;
    const fs::path input = fs::absolute(requested).lexically_normal();
    const bool alias = fs::is_symlink(input);
    // Resolve parent-directory aliases as well; otherwise an explicitly named
    // generation through a symlinked storage directory could bypass validation.
    const fs::path payload = fs::canonical(input);
    if (!alias && !checkpoint_detail::is_generation(payload)) return payload.string();
    if (!checkpoint_detail::is_generation(payload) || payload.filename() != "checkpoint.bin")
        throw std::runtime_error("Invalid checkpoint generation alias: " + requested);
    std::ifstream manifest(payload.parent_path() / "COMPLETE");
    std::string magic;
    unsigned version = 0;
    size_t count = 0;
    if (!(manifest >> magic >> version >> count) || magic != "GRANIT_CHECKPOINT" || version != 1 || count == 0)
        throw std::runtime_error("Incomplete checkpoint generation: " + requested);
    bool has_payload = false;
    std::vector<std::string> names;
    for (size_t i = 0; i < count; ++i) {
        std::string name;
        uintmax_t size = 0;
        uint64_t hash = 0;
        if (!(manifest >> name >> size >> hash) || !checkpoint_detail::valid_member_name(name))
            throw std::runtime_error("Invalid checkpoint manifest: " + requested);
        for (const auto& seen : names) if (name == seen)
            throw std::runtime_error("Duplicate checkpoint manifest member: " + name);
        names.push_back(name);
        const fs::path member = payload.parent_path() / name;
        if (fs::is_symlink(member) || !fs::is_regular_file(member) || fs::file_size(member) != size ||
            checkpoint_detail::checksum(member) != hash)
            throw std::runtime_error("Checkpoint member is missing or damaged: " + member.string());
        has_payload = has_payload || name == "checkpoint.bin";
    }
    manifest >> std::ws;
    if (!has_payload || !manifest.eof()) throw std::runtime_error("Invalid checkpoint manifest: " + requested);
    // An unlisted sidecar would evade the generation's integrity contract.
    for (const auto& entry : fs::directory_iterator(payload.parent_path())) {
        const std::string name = entry.path().filename().string();
        if (name == "COMPLETE") continue;
        bool listed = false;
        for (const auto& member : names) listed = listed || member == name;
        if (!listed) throw std::runtime_error("Unlisted checkpoint member: " + entry.path().string());
    }
    return payload.string();
}

class CheckpointTransaction {
    std::filesystem::path final_, generation_, publish_;
    bool published_ = false;
public:
    explicit CheckpointTransaction(const std::string& final_path)
        : final_(std::filesystem::absolute(final_path).lexically_normal()) {
        std::filesystem::create_directories(final_.parent_path());
        std::string pattern = (final_.parent_path() / ".checkpoint-generation-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        const char* created = ::mkdtemp(name.data());
        if (!created) throw std::runtime_error("Cannot create checkpoint generation: " + std::string(std::strerror(errno)));
        generation_ = created;
        publish_ = final_.parent_path() / (generation_.filename().string() + ".publish");
    }
    CheckpointTransaction(const CheckpointTransaction&) = delete;
    CheckpointTransaction& operator=(const CheckpointTransaction&) = delete;
    ~CheckpointTransaction() {
        std::error_code ignored;
        std::filesystem::remove(publish_, ignored);
        if (!published_) std::filesystem::remove_all(generation_, ignored);
    }
    std::string staging_path() const { return (generation_ / "checkpoint.bin").string(); }
    bool published() const { return published_; }
    void commit() {
        namespace fs = std::filesystem;
        if (published_) throw std::runtime_error("Checkpoint transaction was already published");
        std::vector<fs::path> members;
        bool has_payload = false;
        for (const auto& entry : fs::directory_iterator(generation_)) {
            const std::string name = entry.path().filename().string();
            if (!checkpoint_detail::valid_member_name(name) || (name.size() >= 4 && name.substr(name.size()-4) == ".tmp") ||
                entry.is_symlink() || !entry.is_regular_file())
                throw std::runtime_error("Unfinished or invalid checkpoint member: " + entry.path().string());
            checkpoint_detail::sync_path(entry.path());
            members.push_back(entry.path());
            has_payload = has_payload || name == "checkpoint.bin";
        }
        if (!has_payload) throw std::runtime_error("Checkpoint payload is missing");
        const fs::path manifest_path = generation_ / "COMPLETE";
        std::ofstream manifest(manifest_path, std::ios::trunc);
        manifest << "GRANIT_CHECKPOINT 1 " << members.size() << '\n';
        for (const auto& member : members)
            manifest << member.filename().string() << ' ' << fs::file_size(member) << ' '
                     << checkpoint_detail::checksum(member) << '\n';
        manifest.flush();
        manifest.close();
        if (!manifest) throw std::runtime_error("Cannot write checkpoint manifest: " + manifest_path.string());
        checkpoint_detail::sync_path(manifest_path);
        checkpoint_detail::sync_path(generation_, true);
        fs::create_symlink(generation_.filename() / "checkpoint.bin", publish_);
        fs::rename(publish_, final_); // The only publication point; old generation remains valid.
        published_ = true;
        try { checkpoint_detail::sync_path(final_.parent_path(), true); }
        catch (const std::exception& error) {
            // Publication already happened. Never delete the visible generation.
            throw std::runtime_error(std::string("Checkpoint published, but directory durability is uncertain: ") + error.what());
        }
    }
};

} // namespace cudadem
#endif
