/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <CL/sycl.hpp>
#include <common/EventBase.hpp>
#include <af/event.h>

namespace oneapi {
class OneAPIEventPolicy {
   public:
    using EventType = sycl::event;
    using QueueType = sycl::queue;
    //using ErrorType = sycl::exception; //does this make sense
    using ErrorType = void;

    static ErrorType createAndMarkEvent(cl_event *e) noexcept {
        // Events are created when you mark them
        //return CL_SUCCESS;
    }

    static ErrorType markEvent(cl_event *e, cl_command_queue stream) noexcept {
        //return clEnqueueMarkerWithWaitList(stream, 0, nullptr, e);
    }

    static ErrorType waitForEvent(cl_event *e, cl_command_queue stream) noexcept {
        //return clEnqueueMarkerWithWaitList(stream, 1, e, nullptr);
    }

    static ErrorType syncForEvent(cl_event *e) noexcept {
        //return clWaitForEvents(1, e);
    }

    static ErrorType destroyEvent(cl_event *e) noexcept {
        //return clReleaseEvent(*e);
    }
};

using Event = common::EventBase<OneAPIEventPolicy>;

/// \brief Creates a new event and marks it in the queue
/*
Event makeEvent(cl::CommandQueue &queue);

af_event createEvent();

void markEventOnActiveQueue(af_event eventHandle);

void enqueueWaitOnActiveQueue(af_event eventHandle);

void block(af_event eventHandle);

af_event createAndMarkEvent();
*/

}  // namespace oneapi
