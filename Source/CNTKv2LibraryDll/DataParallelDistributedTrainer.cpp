//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "DataParallelDistributedTrainer.h"

namespace CNTK
{
    DistributedTrainerPtr CreateDataParallelDistributedTrainer(DistributedCommunicatorPtr communicator, bool useAsyncBufferedParameterUpdate)
    {
        return std::make_shared<DataParallelDistributedTrainer>(communicator, useAsyncBufferedParameterUpdate);
    }

    DataParallelDistributedTrainer::DataParallelDistributedTrainer(DistributedCommunicatorPtr communicator, bool useAsyncBufferedParameterUpdate)
        : m_communicator(communicator),
        m_useAsyncBufferedParameterUpdate(useAsyncBufferedParameterUpdate)
    {
        if (!useAsyncBufferedParameterUpdate)
            LogicError("Asynchronous parameter update is not yet supported.");
    }

    // Optional override that gets called per minibatch after finishing gradient computation but before updating model parameters
    void DataParallelDistributedTrainer::PreParameterUpdateCallback(const Trainer& /*trainer*/, const std::unordered_map<Variable, ValuePtr>& /*gradientValues*/, const MinibatchInfo& /*info*/)
    {
        NOT_IMPLEMENTED;
    }

    // Optional override that gets called before each minbatch during training
    void DataParallelDistributedTrainer::PreMinibatchCallback(const Trainer& /*trainer*/)
    {
    }

    // Optionally overridable method to get checkpoint state associated with this Distributed train method
    Dictionary DataParallelDistributedTrainer::GetCheckpointState() const
    {
        // Currently we do not safe the state of the distributed trainer.
        return Dictionary();
    }

    // Optionally overridable method to restore state pertaining this distributed training method from a previous checkpoint
    void DataParallelDistributedTrainer::RestoreFromCheckpoint(const Dictionary& /*checkpoint*/)
    {
        // Currently we do not safe the state of the distributed trainer.
    }
}