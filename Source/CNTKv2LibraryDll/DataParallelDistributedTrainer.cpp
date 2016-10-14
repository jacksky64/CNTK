//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "DataParallelDistributedTrainer.h"

namespace CNTK
{
    // Optional override that gets called per minibatch after finishing gradient computation but before updating model parameters
    void DataParallelDistributedTrainer::PreParameterUpdateCallback(const Trainer& /*trainer*/, const std::unordered_map<Variable, Value>& /*gradientValues*/)
    {
        NOT_IMPLEMENTED;
    }

    // Optional override that gets called before each minbatch during training
    void DataParallelDistributedTrainer::PreMinibatchCallback(const Trainer& /*trainer*/)
    {
        NOT_IMPLEMENTED;
    }

    // Optionally overridable method to get checkpoint state associated with this Distributed train method
    Dictionary DataParallelDistributedTrainer::GetCheckpointState() const
    {
        NOT_IMPLEMENTED;
    }

    // Optionally overridable method to restore state pertaining this distributed training method from a previous checkpoint
    void DataParallelDistributedTrainer::RestoreFromCheckpoint(const Dictionary& /*checkpoint*/)
    {
        NOT_IMPLEMENTED;
    }
}