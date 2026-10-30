#include "src/core/constant/checkpoint_transaction.h"
#include <iostream>
#include <sys/wait.h>

namespace fs = std::filesystem;
using cudadem::CheckpointTransaction;
using cudadem::resolve_checkpoint_path;

static void require(bool condition) {
    if (!condition) throw std::runtime_error("Checkpoint transaction test assertion failed");
}

static void write(const std::string& path, const std::string& value) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << value;
    out.close();
    require(static_cast<bool>(out));
}
static std::string read(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
template<class F> static void rejects(F function) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    require(rejected);
}

int main() {
    char directory[] = "/tmp/granit-checkpoint-transaction-XXXXXX";
    require(::mkdtemp(directory));
    const fs::path root(directory);
    const std::string alias = (root / "checkpoint_step_0000000042.bin").string();
    // Regular v3 files predate transaction manifests and remain loadable.
    write(alias, "legacy");
    require(read(resolve_checkpoint_path(alias)) == "legacy");
    {
        CheckpointTransaction tx(alias);
        write(tx.staging_path(), "old-particles");
        write(tx.staging_path() + ".history", "old-springs");
        require(read(alias) == "legacy");
        tx.commit();
        require(tx.published());
    }
    const std::string old = resolve_checkpoint_path(alias);
    require(fs::is_symlink(alias));
    require(read(old) == "old-particles" && read(old + ".history") == "old-springs");
    // A real process dies after payload creation, before sidecars/publication.
    const pid_t child = ::fork();
    require(child >= 0);
    if (child == 0) {
        CheckpointTransaction tx(alias);
        write(tx.staging_path(), "unpublished-particles");
        ::_exit(23);
    }
    int status = 0;
    require(::waitpid(child, &status, 0) == child);
    require(WIFEXITED(status) && WEXITSTATUS(status) == 23);
    require(resolve_checkpoint_path(alias) == old);
    // Discovery sees only the published .bin alias, never a crashed generation.
    size_t visible_checkpoints = 0;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (entry.path().extension() == ".bin") ++visible_checkpoints;
        if (entry.is_directory() && !fs::exists(entry.path() / "COMPLETE"))
            rejects([&] { resolve_checkpoint_path((entry.path() / "checkpoint.bin").string()); });
    }
    require(visible_checkpoints == 1);
    // Failed staging is harmless even when replacing the exact same step.
    {
        CheckpointTransaction tx(alias);
        write(tx.staging_path(), "bad");
        write(tx.staging_path() + ".history.tmp", "unfinished");
        rejects([&] { tx.commit(); });
    }
    require(resolve_checkpoint_path(alias) == old);
    // Successful replacement atomically changes payload and every sidecar.
    {
        CheckpointTransaction tx(alias);
        write(tx.staging_path(), "new-particles");
        write(tx.staging_path() + ".history", "new-springs");
        tx.commit();
    }
    const std::string current = resolve_checkpoint_path(alias);
    require(current != old && read(current) == "new-particles");
    require(read(current + ".history") == "new-springs");
    // A reader that resolved before replacement still owns a complete bundle.
    require(read(old) == "old-particles" && read(old + ".history") == "old-springs");
    fs::create_directory_symlink(fs::path(current).parent_path(), root / "generation-directory-alias");
    const std::string through_parent_alias = (root / "generation-directory-alias" / "checkpoint.bin").string();
    require(resolve_checkpoint_path(through_parent_alias) == current);
    // Same-size corruption must be rejected, not merely checked for existence.
    write(current + ".history", "bad-springs");
    rejects([&] { resolve_checkpoint_path(alias); });
    rejects([&] { resolve_checkpoint_path(current); });
    rejects([&] { resolve_checkpoint_path(through_parent_alias); });
    write(current + ".history", "new-springs");
    require(resolve_checkpoint_path(alias) == current);
    fs::remove(current + ".history");
    rejects([&] { resolve_checkpoint_path(alias); });
    write(current + ".history", "new-springs");
    // A concurrent reader resolves once and must never mix two generations.
    const pid_t reader = ::fork();
    require(reader >= 0);
    if (reader == 0) {
        try {
            for (int i = 0; i < 1000; ++i) {
                const std::string payload = resolve_checkpoint_path(alias);
                const std::string value = read(payload);
                const std::string history = read(payload + ".history");
                if (value == "new-particles" ? history != "new-springs" : history != value + "-springs")
                    ::_exit(90);
            }
        } catch (...) { ::_exit(91); }
        ::_exit(0);
    }
    for (int i = 0; i < 20; ++i) {
        CheckpointTransaction tx(alias);
        const std::string value = std::to_string(i);
        write(tx.staging_path(), value);
        write(tx.staging_path() + ".history", value + "-springs");
        tx.commit();
    }
    require(::waitpid(reader, &status, 0) == reader);
    require(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    fs::remove_all(root);
    std::cout << "checkpoint transaction tests passed\n";
}
