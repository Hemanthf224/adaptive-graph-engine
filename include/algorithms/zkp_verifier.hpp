#pragma once
#include "core/graph.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace graph_engine {
namespace algorithms {

// A Pedersen-style commitment to a vector of PageRank scores
struct ZKPCommitment {
    uint64_t commitment_hash; // Hash of the scores + randomness (simulates Pedersen commitment)
    uint64_t randomness;      // Blinding factor (secret)
    size_t num_vertices;
};

// A non-interactive proof that PageRank was computed correctly
struct ZKPProof {
    uint64_t proof_hash;     // Hash-based proof (Fiat-Shamir heuristic)
    bool is_valid;
};

class ZKPVerifier {
public:
    // Prover: Commit to a set of PageRank scores WITHOUT revealing them
    static ZKPCommitment commit(const std::vector<float>& scores, uint64_t randomness = 0xDEADBEEFCAFEBABE);

    // Prover: Generate a proof that the committed scores were computed from the given graph
    static ZKPProof prove(const core::CSRGraph& graph, const ZKPCommitment& commitment, const std::vector<float>& scores);

    // Verifier: Verify the proof WITHOUT seeing the original scores
    static bool verify(const core::CSRGraph& graph, const ZKPCommitment& commitment, const ZKPProof& proof);
};

} // namespace algorithms
} // namespace graph_engine
