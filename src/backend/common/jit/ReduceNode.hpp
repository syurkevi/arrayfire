/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/jit/NaryNode.hpp>
#include <jit/kernel_generators.hpp>

#include <cmath>

namespace common {
class ReduceNode : public NaryNode {
  protected:
      int m_axis;
      af::dim4 m_input_size;
  public:
    ReduceNode(const char *out_type_str, const char *name_str,
               common::Node_ptr in, int axis, int op, af::dim4 input_size)
        : NaryNode(out_type_str, name_str, "__noop", 1, {{in}}, op, in->getHeight() + 1)
        , m_input_size(input_size)
        , m_axis(axis) {}

    /// Generates the code for writing output to global memory
    ///
    /// Generates the source code to perform a global memory write
    /// for the reduce node.
    ///
    /// \param[in/out] kerStream  The string will be written to this stream
    /// \param[in]     id         The integer id of the node
    void genGlobalWrite(std::stringstream &kerStream,
                        const int &id) const final {
        detail::generateReduceNodeRead(kerStream, id, m_input_size, m_axis);
    }

    af::dim4 getInputSize() { return m_input_size; }
    bool isLinear(dim_t dims[4]) const final{
        UNUSED(dims);
        return m_axis == 0 || m_axis == 1;
    }
};
}  // namespace common
