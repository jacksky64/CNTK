//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "Function.h"

namespace CNTK
{
    Trainer::Trainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const FunctionPtr& evaluationFunction, const std::vector<LearnerPtr>& parameterLearners)
        : m_model(model), m_lossFunction(lossFunction), m_evaluationFunction(evaluationFunction), m_parameterLearners(parameterLearners), m_prevMinibatchNumSamples(1)
    {
        if (m_lossFunction->Output().DynamicAxes().empty())
            InvalidArgument("The loss function specified in the Trainer constructor must correspond to minibatch data and have dynamic axes");

        if (m_evaluationFunction && m_evaluationFunction->Output().DynamicAxes().empty())
            InvalidArgument("The evaluation function specified in the Trainer constructor must correspond to minibatch data and have dynamic axes");

        m_aggregatedLossFunction = ReduceSum(lossFunction);
        if (m_evaluationFunction)
            m_aggregatedEvaluationFunction = ReduceSum(m_evaluationFunction);

        std::vector<Variable> combinedFunctionArgs = { m_model, m_aggregatedLossFunction, m_lossFunction };
        if (m_evaluationFunction)
        {
            combinedFunctionArgs.push_back(m_aggregatedEvaluationFunction);
            combinedFunctionArgs.push_back(m_evaluationFunction);
        }

        m_combinedTrainingFunction = Combine(combinedFunctionArgs);

        auto modelParameters = m_combinedTrainingFunction->Parameters();
        std::unordered_set<Parameter> learnerParameters;
        for (const auto& learner : parameterLearners)
        {
            const auto& currentLearnerParameters = learner->Parameters();
            for (const auto& parameter : currentLearnerParameters)
            {
                auto insertRetVal = learnerParameters.insert(parameter);
                if (!insertRetVal.second)
                    InvalidArgument("Trainer ctor: Parameter named %S is covered by 2 different learners", parameter.Name().c_str());
            }
        }

        std::unordered_set<Parameter> modelParametersSet(modelParameters.begin(), modelParameters.end());
        if (modelParametersSet != learnerParameters)
        {
            InvalidArgument("Trainer ctor: Union of the parameters covered by the specified parameterLearners should match the specified model's parameters");
        }
            
    }

    Trainer::Trainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const std::vector<LearnerPtr>& parameterLearners)
        : Trainer(model, lossFunction, nullptr, parameterLearners)
    {}

    static double GetScalarValue(const ValuePtr& value)
    {
        if (value->Mask())
            LogicError("Scalar Value object cannot have an associated mask");

        auto scalarData = value->Data();
        if (scalarData->Shape().TotalSize() != 1)
            LogicError("Scalar Value object's has a size > 1");

        double scalar = std::numeric_limits<double>::quiet_NaN();
        NDArrayViewPtr cpuData;
        if (scalarData->Device() == DeviceDescriptor::CPUDevice())
            cpuData = scalarData;
        else
        {
            cpuData = std::make_shared<NDArrayView>(scalarData->GetDataType(), scalarData->Shape(), CNTK::DeviceDescriptor::CPUDevice());
            cpuData->CopyFrom(*scalarData);
        }

        if (scalarData->GetDataType() == DataType::Float)
            scalar = *(cpuData->DataBuffer<float>());
        else if (scalarData->GetDataType() == DataType::Double)
            scalar = *(cpuData->DataBuffer<double>());
        else
            LogicError("Unsupported DataType of training loss value");

        return scalar;
    }

    static size_t GetSampleCount(const Variable& var, const ValuePtr& value)
    {
        auto valueDataShape = value->Shape();
        size_t numMaskedSamples = value->MaskedCount();
        size_t numSamplesInDataArrayView = valueDataShape.SubShape(var.Shape().Rank()).TotalSize();
        if (numMaskedSamples > numSamplesInDataArrayView)
            LogicError("Number of masked values cannot exceed the number of samples that the Value object's Data NDArrayView can hold");

        return (numSamplesInDataArrayView - numMaskedSamples);
    }

    double Trainer::TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        if (!m_aggregatedEvaluationFunction)
            InvalidArgument("Trainer::TestMinibatch: Cannot test when no evaluation function was specified during 'this' trainer's construction");

        // TODO: Should we refactor this code that is somewhat similar to the prologue of the TrainMinibatch function
        std::unordered_map<Variable, ValuePtr> outputs = { { m_aggregatedEvaluationFunction, nullptr }, {m_evaluationFunction, nullptr} };
        m_combinedTrainingFunction->Forward(arguments, outputs, computeDevice);

        auto sampleCount = GetSampleCount(m_evaluationFunction, outputs[m_evaluationFunction]);
        return (GetScalarValue(outputs[m_aggregatedEvaluationFunction]) / sampleCount);
    }

    bool Trainer::TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        std::unordered_map<Variable, ValuePtr> outputsToFetch = {};
        return TrainMinibatch(arguments, outputsToFetch, computeDevice);
    }

    bool Trainer::TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        std::unordered_map<Variable, ValuePtr> outputs = { { m_aggregatedLossFunction, nullptr }, { m_lossFunction, nullptr } };
        if (m_aggregatedEvaluationFunction)
            outputs.insert({ m_aggregatedEvaluationFunction, nullptr });

        outputs.insert(outputsToFetch.begin(), outputsToFetch.end());

        auto backPropSate = m_combinedTrainingFunction->Forward(arguments, outputs, computeDevice, { m_aggregatedLossFunction });
        m_prevMinibatchAggregateTrainingLossValue = outputs[m_aggregatedLossFunction];
        if (m_aggregatedEvaluationFunction)
            m_prevMinibatchAggregateEvalCriterionValue = outputs[m_aggregatedEvaluationFunction];

        for (auto outputToFetch : outputsToFetch)
        {
            if (outputToFetch.second == nullptr)
                outputsToFetch[outputToFetch.first] = outputs[outputToFetch.first];
        }

        ValuePtr rootGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(m_aggregatedLossFunction->Output().GetDataType(), m_prevMinibatchAggregateTrainingLossValue->Shape(), computeDevice), outputs.at(m_aggregatedLossFunction)->Mask());
        if (m_aggregatedLossFunction->Output().GetDataType() == DataType::Float)
            rootGradientValue->Data()->SetValue(1.0f);
        else
            rootGradientValue->Data()->SetValue(1.0);

        auto modelParameters = m_combinedTrainingFunction->Parameters();
        // Since Variable hash function depends on the memory address pointed to by the m_dataFields,
        // after restoring from a checkpoint, when model parameters created fresh,
        // model parameters and learners' parameters no longer hash to the same value,
        //  we need to use uid map here to be able to match parameters.
        std::unordered_map<std::wstring, Variable> uidToParameterMap;
        std::unordered_map<Variable, ValuePtr> parameterGradients;
        for (const auto& parameter : modelParameters)
        {
            parameterGradients[parameter] = nullptr;
            uidToParameterMap[parameter.Uid()] = parameter;
        }

        m_combinedTrainingFunction->Backward(backPropSate, { { m_aggregatedLossFunction, rootGradientValue } }, parameterGradients);

        m_prevMinibatchNumSamples = GetSampleCount(m_lossFunction, outputs[m_lossFunction]);


        bool anyUpdatesPerformed = false;
        for (auto learner : m_parameterLearners)
        {
            std::unordered_map<Parameter, NDArrayViewPtr> learnerParameterGradients;
            const auto& learnerParameters = learner->Parameters();
            for (const auto& learnerParameter : learnerParameters)
            {
                const auto& modelParameter = uidToParameterMap[learnerParameter.Uid()];
                learnerParameterGradients[learnerParameter] = parameterGradients[modelParameter]->Data();

                if (parameterGradients[modelParameter]->Mask())
                    LogicError("The gradient value for a Parameter cannot have an associated mask!");
            }

            anyUpdatesPerformed |= learner->Update(learnerParameterGradients, m_prevMinibatchNumSamples);
        }

        return anyUpdatesPerformed;
    }

    static std::wstring GetTrainerStateCheckpointFilePath(const std::wstring& modelFilePath)
    {
        const wchar_t* checkpointExt = L".ckp";
        return modelFilePath + checkpointExt;
    }

    std::shared_ptr<std::fstream> GetFstream(const std::wstring& filePath, bool readOnly)
    {
        std::ios_base::openmode mode = std::ios_base::binary | (readOnly ? std::ios_base::in : std::ios_base::out);
#ifdef _MSC_VER
        return std::make_shared<std::fstream>(filePath, mode);
#else
        return std::make_shared<std::fstream>(wtocharpath(filePath.c_str()).c_str(), mode);
#endif
    }

    void Trainer::SaveCheckpoint(const std::wstring& modelFilePath, bool usinglegacyModelFormat)
    {
        if (usinglegacyModelFormat)
        {
            SaveAsLegacyModel(m_combinedTrainingFunction, modelFilePath);
        }
        else
        {
             Dictionary model = Function::Save(m_combinedTrainingFunction);
             auto stream = GetFstream(modelFilePath, false);
            *stream << model;
             stream->flush();
        }

        vector<DictionaryValue> learnerStates;

        for (const auto& learner : m_parameterLearners)
        {
            // TODO: add DictionaryValue(T&&)
            learnerStates.push_back(DictionaryValue(learner->Serialize()));
        }
        
        std::wstring trainerStateCheckpointFilePath = GetTrainerStateCheckpointFilePath(modelFilePath);
        auto ckpStream = GetFstream(trainerStateCheckpointFilePath, false);
        // TODO: this will create an extra copy of all leaner states, 
        // add DictionaryValue ctor that takes an rvalue!
        *ckpStream << DictionaryValue(learnerStates);
        ckpStream->flush();
    }

    void Trainer::RestoreFromCheckpoint(const std::wstring& modelFilePath, bool usinglegacyModelFormat)
    {
        // Restore the model's parameters
        if (usinglegacyModelFormat)
        {
            m_combinedTrainingFunction->RestoreFromLegacyModel(modelFilePath);
        }
        else
        { 
            auto stream = GetFstream(modelFilePath, true);
            Dictionary model;
            *stream >> model;
            auto reloadedFunction = Function::Load(model, DeviceDescriptor::CPUDevice());

            std::unordered_map<std::wstring, Variable> inputMap;
            const auto& inputs = m_combinedTrainingFunction->Inputs();
            for (const auto& input : inputs)
            {
                if (input.IsInput())
                {
                    inputMap[input.Uid()] = input;
                }
            }

            std::unordered_map<Variable, Variable> replacements;
            const auto& reloadedFunctionInputs = reloadedFunction->Inputs();
            for (const auto& input : reloadedFunctionInputs)
            {
                if (input.IsPlaceholder())
                {
                    const auto& it = inputMap.find(input.Uid());
                    assert(it != inputMap.end());
                    replacements[input] = it->second;
                }
            }

            m_combinedTrainingFunction = reloadedFunction->ReplacePlaceholders(replacements);
            auto outputs = m_combinedTrainingFunction->Outputs();
            m_model = outputs[0].Owner();
            m_aggregatedLossFunction = outputs[1].Owner();
            m_lossFunction = outputs[2].Owner();
            if (outputs.size() > 3)
            {
                m_aggregatedEvaluationFunction = outputs[3].Owner();
                m_evaluationFunction = outputs[4].Owner();
            }
        }

        std::wstring trainerStateCheckpointFilePath = GetTrainerStateCheckpointFilePath(modelFilePath);
        auto ckpStream = GetFstream(trainerStateCheckpointFilePath, true);
        DictionaryValue checkpoint;
        *ckpStream >> checkpoint;

        const vector<DictionaryValue>& learnerStates = checkpoint.Value<vector<DictionaryValue>>();

        if (learnerStates.size() != m_parameterLearners.size())
        {
            LogicError("Trainer::RestoreFromCheckpoint: "
                       "Number of learners in the checkpoint (%d) does not match the expected number (%d)",
                       learnerStates.size(), m_parameterLearners.size());
        }

        std::unordered_map<std::wstring, Variable> uidToParameterMap;
        const auto& modelParameters = m_combinedTrainingFunction->Parameters();
        for (const auto& parameter : modelParameters)
        {
            uidToParameterMap[parameter.Uid()] = parameter;
        }

        std::unordered_set<std::wstring> learnerParameterUids;
        for (int i = 0; i < m_parameterLearners.size(); ++i)
        {
            m_parameterLearners[i]->RestoreFromCheckpoint(learnerStates[i].Value<Dictionary>());
            auto& learnerParameters = m_parameterLearners[i]->Parameters(); // returns const set.
            for (const auto& parameter : learnerParameters) 
            {
                //auto& p = const_cast<Parameter&>(parameter);
                auto& modelParameter = uidToParameterMap[parameter.Uid()];
                auto value = reinterpret_cast<Parameter&>(modelParameter).Value();
                parameter.m_dataFields->m_value = value;
                learnerParameterUids.insert(parameter.Uid());
            }
        }

        if (modelParameters.size() != learnerParameterUids.size())
        {
            LogicError("Trainer::RestoreFromCheckpoint: "
                       "Number of unique model parameters (%d) and does not match the number of unique learner parameters (%d)",
                       modelParameters.size(), learnerParameterUids.size());
        }

        for (const auto& p : modelParameters) 
        {
            if (learnerParameterUids.find(p.Uid()) == learnerParameterUids.end())
            {
                LogicError("Trainer::RestoreFromCheckpoint: "
                           "No learner is associated with the parameter %ls (uid: %ls)", p.Name().c_str(), p.Uid().c_str());
            }
        }
    }

    double Trainer::PreviousMinibatchLossAverage() const
    {
        return (GetScalarValue(m_prevMinibatchAggregateTrainingLossValue) / m_prevMinibatchNumSamples);
    }

    double Trainer::PreviousMinibatchEvaluationAverage() const
    {
        if (!m_evaluationFunction)
            InvalidArgument("Trainer::PreviousMinibatchEvaluationAverage: Cannot get evaluation criterion value when no evaluation function was specified during 'this' trainer's construction");

        return (GetScalarValue(m_prevMinibatchAggregateEvalCriterionValue) / m_prevMinibatchNumSamples);
    }
}
