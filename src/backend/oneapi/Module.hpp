/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/ModuleInterface.hpp>
#include <CL/sycl.hpp>


namespace oneapi {

/// oneapi backend wrapper for cl::Program object
class Module : public common::ModuleInterface<sycl::program> {
   public:
    using ModuleType = sycl::program;
    using BaseClass  = common::ModuleInterface<ModuleType>;

    /// \brief Create an uninitialized Module
    Module() = default;

    /// \brief Create a module given a sycl::program type
    Module(ModuleType mod) : BaseClass(mod) {}

    /// \brief Unload module
    operator bool() const final { return get()(); }

    /// Unload the module
    void unload() final { set(sycl::program()); }
};

}  // namespace oneapi
