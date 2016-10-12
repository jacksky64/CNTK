//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "CNTKLibrary.h"
#include "Common.h"
#include <string>
#include <random>
#include <vector>
#include <functional>
#include <iostream>

#define CAT(A, B)   A##B
#define WSTRING(A)  CAT(L, #A)


using namespace CNTK;
using namespace std;

using namespace Microsoft::MSR::CNTK;

static const size_t maxNestingDepth = 10;
static const size_t maxNestedDictSize = 10;
static const size_t maxNestedVectorSize = 100;
static const size_t maxNDShapeSize = 100;

static const size_t maxNumAxes = 10;
static const size_t maxDimSize = 15;


static size_t keyCounter = 0;
static uniform_real_distribution<double> double_dist = uniform_real_distribution<double>();
static uniform_real_distribution<float> float_dist = uniform_real_distribution<float>();

static std::wstring tempFilePath = L"serialization.tmp";

DictionaryValue CreateDictionaryValue(DictionaryValue::Type, size_t);

DictionaryValue::Type GetType()
{
    return DictionaryValue::Type(rng() % (unsigned int) DictionaryValue::Type::NDArrayView + 1);
}

void AddKeyValuePair(Dictionary& dict, size_t depth)
{
    auto type = GetType();
    while (depth >= maxNestingDepth && 
           type == DictionaryValue::Type::Vector ||
           type == DictionaryValue::Type::Dictionary)
    {
        type = GetType();
    }
    dict[L"key" + to_wstring(keyCounter++)] = CreateDictionaryValue(type, depth);
}

Dictionary CreateDictionary(size_t size, size_t depth = 0) 
{
    Dictionary dict;
    for (auto i = 0; i < size; ++i)
    {
        AddKeyValuePair(dict, depth);
    }

    return dict;
}

template <typename ElementType>
NDArrayViewPtr CreateNDArrayView(size_t numAxes, const DeviceDescriptor& device) 
{
    NDShape viewShape(numAxes);
    for (size_t i = 0; i < numAxes; ++i)
        viewShape[i] = (rng() % maxDimSize) + 1;

    return NDArrayView::RandomUniform<ElementType>(viewShape, ElementType(-4.0), ElementType(19.0), 1, device);
}

NDArrayViewPtr CreateNDArrayView()
{
    auto numAxes = (rng() % maxNumAxes) + 1;
    auto device = DeviceDescriptor::CPUDevice();
#ifndef CPUONLY
    if (rng() % 2 == 0)
    {
        device = DeviceDescriptor::GPUDevice(0);
    }
#endif

    return (rng() % 2 == 0) ? 
        CreateNDArrayView<float>(numAxes, device) : CreateNDArrayView<double>(numAxes, device);
}

DictionaryValue CreateDictionaryValue(DictionaryValue::Type type, size_t depth)
{
    switch (type)
    {
    case DictionaryValue::Type::Bool:
        return DictionaryValue(!!(rng() % 2));
    case DictionaryValue::Type::SizeT:
        return DictionaryValue(rng());
    case DictionaryValue::Type::Float:
        return DictionaryValue(float_dist(rng));
    case DictionaryValue::Type::Double:
        return DictionaryValue(double_dist(rng));
    case DictionaryValue::Type::String:
        return DictionaryValue(to_wstring(rng()));
    case DictionaryValue::Type::Axis:
        return ((rng() % 2) == 0) ? DictionaryValue(Axis(0)) : DictionaryValue(Axis(L"newDynamicAxis_" + to_wstring(rng())));
    case DictionaryValue::Type::NDShape:
    {
        size_t size = rng() % maxNDShapeSize + 1;
        NDShape shape(size);
        for (auto i = 0; i < size; i++)
        {
            shape[i] = rng();
        }
        return DictionaryValue(shape);
    }
    case DictionaryValue::Type::Vector:
    {   
        auto type = GetType();
        size_t size = rng() % maxNestedVectorSize + 1;
        vector<DictionaryValue> vector(size);
        for (auto i = 0; i < size; i++)
        {
            vector[i] = CreateDictionaryValue(type, depth + 1);
        }
        return DictionaryValue(vector);
    }
    case DictionaryValue::Type::Dictionary:
        return DictionaryValue(CreateDictionary(rng() % maxNestedDictSize  + 1, depth + 1));
    case DictionaryValue::Type::NDArrayView:
        return DictionaryValue(*(CreateNDArrayView()));
    default:
        NOT_IMPLEMENTED;
    }
}

void TestDictionarySerialization(size_t dictSize) 
{
    if ((_wunlink(tempFilePath.c_str()) != 0) && (errno != ENOENT))
        std::runtime_error("Error deleting temporary test file 'serialization.tmp'.");

    Dictionary originalDict = CreateDictionary(dictSize);
    
    {
        fstream stream;
        OpenStream(stream, tempFilePath, false);
        stream << originalDict;
        stream.flush();
    }

    Dictionary deserializedDict;

    {
        fstream stream;
        OpenStream(stream, tempFilePath, true);
        stream >> deserializedDict;
    }
    
    if (originalDict != deserializedDict)
        throw std::runtime_error("TestDictionarySerialization: original and deserialized dictionaries are not identical.");
}

template <typename ElementType>
void TestLearnerSerialization(int numParameters, const DeviceDescriptor& device) 
{
    if ((_wunlink(tempFilePath.c_str()) != 0) && (errno != ENOENT))
        std::runtime_error("Error deleting temporary test file 'serialization.tmp'.");

    NDShape shape = CreateShape(5, maxDimSize);

    vector<Parameter> parameters;
    unordered_map<Parameter, NDArrayViewPtr> gradientValues;
    for (int i = 0; i < numParameters; i++)
    {
        Parameter parameter(NDArrayView::RandomUniform<ElementType>(shape, -0.5, 0.5, i, device), L"parameter_" + to_wstring(i));
        parameters.push_back(parameter);
        gradientValues[parameter] = NDArrayView::RandomUniform<ElementType>(shape, -0.5, 0.5, numParameters + i, device);
    }

    auto learner1 = SGDLearner(parameters, 0.05);
    
    learner1->Update(gradientValues, 1);

    {
        auto checkpoint = learner1->Serialize();
        fstream stream;
        OpenStream(stream, tempFilePath, false);
        stream << checkpoint;
        stream.flush();
    }

    auto learner2 = SGDLearner(parameters, 0.05);

    {
        Dictionary checkpoint;
        fstream stream;
        OpenStream(stream, tempFilePath, true);
        stream >> checkpoint;
        learner2->RestoreFromCheckpoint(checkpoint);
    }

    int i = 0;
    for (auto parameter : parameters)
    {
        gradientValues[parameter] = NDArrayView::RandomUniform<ElementType>(shape, -0.5, 0.5, 2*numParameters + i, device);
        i++;
    }

    learner1->Update(gradientValues, 1);
    learner2->Update(gradientValues, 1);

     auto checkpoint1 = learner1->Serialize();
     auto checkpoint2 = learner2->Serialize();
    
    if (checkpoint1 != checkpoint2)
        throw std::runtime_error("TestLearnerSerialization: original and restored from a checkpoint learners diverge.");
}


void CheckEnumValuesNotModified() {
    // During the model and checkpoint serialization, for all enum values we save corresponding 
    // integer values. For this reason, we need to make sure that enum values never change 
    // corresponding integer values (new enum values can only be appended to the end of the value
    // list and never inserted in the middle). 

    // The following list of asserts is APPEND ONLY. DO NOT CHANGE existing assert statements.

    
    static_assert(static_cast<size_t>(DataType::Unknown) == 0 &&
                  static_cast<size_t>(DataType::Float) == 1 &&
                  static_cast<size_t>(DataType::Double) == 2, 
                  "DataType enum value was modified.");

    static_assert(static_cast<size_t>(VariableKind::Input) == 0 &&
                  static_cast<size_t>(VariableKind::Output) == 1 &&
                  static_cast<size_t>(VariableKind::Parameter) == 2 &&
                  static_cast<size_t>(VariableKind::Constant) == 3 &&
                  static_cast<size_t>(VariableKind::Placeholder) == 4, 
                  "VariableKind enum value was modified.");

    
    static_assert(static_cast<size_t>(PrimitiveOpType::Negate) == 0 &&
                  static_cast<size_t>(PrimitiveOpType::Sigmoid) == 1 &&
                  static_cast<size_t>(PrimitiveOpType::Tanh) == 2 &&
                  static_cast<size_t>(PrimitiveOpType::ReLU) == 3 &&
                  static_cast<size_t>(PrimitiveOpType::Exp) == 4 &&
                  static_cast<size_t>(PrimitiveOpType::Log) == 5 &&
                  static_cast<size_t>(PrimitiveOpType::Sqrt) == 6 &&
                  static_cast<size_t>(PrimitiveOpType::Floor) == 7 &&
                  static_cast<size_t>(PrimitiveOpType::Abs) == 8 &&
                  static_cast<size_t>(PrimitiveOpType::Reciprocal) == 9 &&
                  static_cast<size_t>(PrimitiveOpType::Softmax) == 10 &&
                  static_cast<size_t>(PrimitiveOpType::Hardmax) == 11 &&
                  static_cast<size_t>(PrimitiveOpType::TransposeAxes) == 12 &&
                  static_cast<size_t>(PrimitiveOpType::Where) == 13 &&
                  static_cast<size_t>(PrimitiveOpType::Slice) == 14 &&
                  static_cast<size_t>(PrimitiveOpType::Dropout) == 15 &&
                  static_cast<size_t>(PrimitiveOpType::Reshape) == 16 &&
                  static_cast<size_t>(PrimitiveOpType::Pooling) == 17 &&
                  static_cast<size_t>(PrimitiveOpType::SumAll) == 18 &&
                  static_cast<size_t>(PrimitiveOpType::Plus) == 19  &&
                  static_cast<size_t>(PrimitiveOpType::Minus) == 20 &&
                  static_cast<size_t>(PrimitiveOpType::ElementTimes) == 21 &&
                  static_cast<size_t>(PrimitiveOpType::Equal) == 22 &&
                  static_cast<size_t>(PrimitiveOpType::NotEqual) == 23 &&
                  static_cast<size_t>(PrimitiveOpType::Less) == 24 &&
                  static_cast<size_t>(PrimitiveOpType::LessEqual) == 25 &&
                  static_cast<size_t>(PrimitiveOpType::Greater) == 26 &&
                  static_cast<size_t>(PrimitiveOpType::GreaterEqual) == 27 &&
                  static_cast<size_t>(PrimitiveOpType::PackedIndex) == 28 &&
                  static_cast<size_t>(PrimitiveOpType::GatherPacked) == 29 &&
                  static_cast<size_t>(PrimitiveOpType::ScatterPacked) == 30 &&
                  static_cast<size_t>(PrimitiveOpType::Times) == 31 &&
                  static_cast<size_t>(PrimitiveOpType::TransposeTimes) == 32 &&
                  static_cast<size_t>(PrimitiveOpType::Convolution) == 33 &&
                  static_cast<size_t>(PrimitiveOpType::SquaredError) == 34 &&
                  static_cast<size_t>(PrimitiveOpType::CrossEntropyWithSoftmax) == 35 &&
                  static_cast<size_t>(PrimitiveOpType::ClassificationError) == 36 &&
                  static_cast<size_t>(PrimitiveOpType::PastValue) == 37 &&
                  static_cast<size_t>(PrimitiveOpType::FutureValue) == 38 &&
                  static_cast<size_t>(PrimitiveOpType::ReduceElements) == 39 &&
                  static_cast<size_t>(PrimitiveOpType::BatchNormalization) == 40 &&
                  static_cast<size_t>(PrimitiveOpType::Clip) == 41 &&
                  static_cast<size_t>(PrimitiveOpType::Select) == 42 &&
                  static_cast<size_t>(PrimitiveOpType::Splice) == 43 &&
                  static_cast<size_t>(PrimitiveOpType::Combine) == 44, 
                  "PrimitiveOpType enum value was modified.");
}

void TestFunctionSaveAndLoad(const FunctionPtr& function, const DeviceDescriptor& device)
{
    Dictionary model1 = Function::Save(function);
    auto reloadedFunction = Function::Load(model1, device);
    Dictionary model2 = Function::Save(reloadedFunction);

    if (model1 != model2)
    {
        throw std::runtime_error("TestLearnerSerialization: original and reloaded models diverge.");
    }
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

 void SaveModel(const FunctionPtr& rootFunction, const std::wstring& modelFile)
 {
    Dictionary model = Function::Save(rootFunction);
    auto stream = GetFstream(modelFile, false);
    *stream << model;
    stream->flush();
 }

  FunctionPtr LoadModel(const std::wstring& modelFile, std::vector<Variable> inputs)
 {
    auto stream = GetFstream(modelFile, true);
    Dictionary modelDict;
    *stream >> modelDict;
    auto model = Function::Load(modelDict, DeviceDescriptor::CPUDevice());

    std::unordered_map<std::wstring, Variable> inputMap;
    for (const auto& input : inputs)
    {
        if (input.IsInput())
        {
            inputMap[input.Uid()] = input;
        }
    }

    std::unordered_map<Variable, Variable> replacements;
    const auto& placeholders = model->Placeholders();
    for (const auto& placeholder : placeholders)
    {
        const auto& it = inputMap.find(placeholder.Uid());
        assert(it != inputMap.end());
        replacements[placeholder] = it->second;
    }

    return model->ReplacePlaceholders(replacements);
 }


  void SaveBoth_V1_and_V2_Models(const FunctionPtr& rootFunction, const std::wstring& prefix)
  {
      SaveAsLegacyModel(rootFunction, prefix + L".v1.out");
      SaveModel(rootFunction, prefix + L".v2.out");
  }

void ComparisonTests(const DeviceDescriptor& device)
{
    const size_t inputDim = 11;
    const size_t cellDim = 3;
    const size_t hiddenDim = 5;
    const size_t embeddingDim = 7;
    const size_t numOutputClasses = 2;

    auto features = InputVariable({ inputDim }, true /*isSparse*/, DataType::Float, L"features");
    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels", { Axis::DefaultBatchAxis() });

    auto classifierOutput = LSTMSequenceClassiferNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");
    SaveAsLegacyModel(classifierOutput, L"throw_away_legacy_model");
    
    Internal::ResetUniqueId();
    auto classifierOutput1 = LSTMSequenceClassiferNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");
    
    Internal::ResetUniqueId();
    auto classifierOutput2 = LSTMSequenceClassiferNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");

    SaveBoth_V1_and_V2_Models(classifierOutput1, WSTRING(classifierOutput1));
    SaveBoth_V1_and_V2_Models(classifierOutput2, WSTRING(classifierOutput2));

    Internal::ResetUniqueId();
    auto classifierOutput1_1 = LoadLegacyModel(DataType::Float, WSTRING(classifierOutput1)  L".v1.out", device);
    
    Internal::ResetUniqueId();
    auto classifierOutput1_2 = LoadModel(WSTRING(classifierOutput1)  L".v2.out", {features});

    Internal::ResetUniqueId();
    auto classifierOutput2_1 = LoadLegacyModel(DataType::Float, WSTRING(classifierOutput2)  L".v1.out", device);

    Internal::ResetUniqueId();
    auto classifierOutput2_2 = LoadModel(WSTRING(classifierOutput2)  L".v2.out", {features});

    Internal::ResetUniqueId();
    SaveBoth_V1_and_V2_Models(classifierOutput1_1, WSTRING(classifierOutput1_1));
    SaveBoth_V1_and_V2_Models(classifierOutput1_2, WSTRING(classifierOutput1_2));
    SaveBoth_V1_and_V2_Models(classifierOutput2_1, WSTRING(classifierOutput2_1));
    SaveBoth_V1_and_V2_Models(classifierOutput2_2, WSTRING(classifierOutput2_2));
}




void TestFunctionSerialization(const DeviceDescriptor& device)
{
    const size_t inputDim = 2000;
    const size_t cellDim = 25;
    const size_t hiddenDim = 25;
    const size_t embeddingDim = 50;
    const size_t numOutputClasses = 5;

    auto features = InputVariable({ inputDim }, true /*isSparse*/, DataType::Float, L"features");
    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels", { Axis::DefaultBatchAxis() });

    auto minibatchSource = TextFormatMinibatchSource(L"Train.ctf", { { L"features", inputDim, true, L"x" }, { L"labels", numOutputClasses, false, L"y" } }, 0);
    auto featureStreamInfo = minibatchSource->StreamInfo(features);
    auto labelStreamInfo = minibatchSource->StreamInfo(labels);

    const size_t minibatchSize = 200;
    auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
    auto actualMBSize = minibatchData[labelStreamInfo].m_numSamples;

    LearningRatesPerSample learningRateSchedule({ { 2, 0.0005 }, { 2, 0.00025 } }, actualMBSize);

    auto classifierOutput = LSTMSequenceClassiferNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");
    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");
    auto prediction = CNTK::ClassificationError(classifierOutput, labels, L"classificationError");
    auto learner = SGDLearner(classifierOutput->Parameters(), learningRateSchedule);
    Trainer trainer(classifierOutput, trainingLoss, prediction, { learner }); // load all statics
    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);

    Internal::ResetUniqueId();
    auto classifierOutput1 = LSTMSequenceClassiferNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");
    auto trainingLoss1 = CNTK::CrossEntropyWithSoftmax(classifierOutput1, labels, L"lossFunction");
    auto prediction1 = CNTK::ClassificationError(classifierOutput1, labels, L"classificationError");
    auto learner1 = SGDLearner(classifierOutput1->Parameters(), learningRateSchedule);
    Trainer trainer1(classifierOutput1, trainingLoss1, prediction1, { learner1 });




    trainer1.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    SaveBoth_V1_and_V2_Models(classifierOutput1, WSTRING(classifierOutput1));

    Dictionary checkpoint = learner1->Serialize();

   /* auto dummy = LoadLegacyModel(DataType::Float, WSTRING(classifierOutput1)  L".v1.out", device);
    SaveModel(dummy, WSTRING(classifierOutput1)  L".reloaded.v2.out");
    auto classifierOutput1_1 = LoadModel(WSTRING(classifierOutput1)  L".reloaded.v2.out", {features});*/

    //auto classifierOutput1_1 = LoadLegacyModel(DataType::Float, WSTRING(classifierOutput1)  L".v1.out", device);
    auto classifierOutput1_1 = LSTMSequenceClassiferNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");
    auto trainingLoss1_1 = CNTK::CrossEntropyWithSoftmax(classifierOutput1_1, labels, L"lossFunction");
    auto prediction1_1 = CNTK::ClassificationError(classifierOutput1_1, labels, L"classificationError");
    auto learner1_1 = SGDLearner(classifierOutput1_1->Parameters(), learningRateSchedule);
    learner1_1->RestoreFromCheckpoint(checkpoint);
    Trainer trainer1_1(classifierOutput1_1, trainingLoss1_1, prediction1_1, { learner1_1 });
    
    auto classifierOutput1_2 = LoadModel(WSTRING(classifierOutput1)  L".v2.out", {features});
    auto trainingLoss1_2 = CNTK::CrossEntropyWithSoftmax(classifierOutput1_2, labels, L"lossFunction");
    auto prediction1_2 = CNTK::ClassificationError(classifierOutput1_2, labels, L"classificationError");
    auto learner1_2 = SGDLearner(classifierOutput1_2->Parameters(), learningRateSchedule);
    learner1_2->RestoreFromCheckpoint(checkpoint);
    Trainer trainer1_2(classifierOutput1_2, trainingLoss1_2, prediction1_2, { learner1_2 });

    //1) this works!
    
    trainer1_1.TrainMinibatch({ { classifierOutput1_1->Arguments()[0], minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    trainer1.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    trainer1_2.TrainMinibatch({ { classifierOutput1_2->Arguments()[0], minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);


    double mbLoss1 = trainer1.PreviousMinibatchLossAverage();
    double mbLoss1_1 = trainer1_1.PreviousMinibatchLossAverage();
    double mbLoss1_2 = trainer1_2.PreviousMinibatchLossAverage();
    if (mbLoss1 != mbLoss1_1 || mbLoss1 != mbLoss1_2)
        throw std::runtime_error("Post checkpoint restoration training loss does not match expectation");

}

void TestModelSerialization(const DeviceDescriptor& device)
{
    const size_t inputDim = 2000;
    const size_t cellDim = 25;
    const size_t hiddenDim = 25;
    const size_t embeddingDim = 50;
    const size_t numOutputClasses = 5;

    auto features = InputVariable({ inputDim }, true /*isSparse*/, DataType::Float, L"features");
    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels", { Axis::DefaultBatchAxis() });

    auto minibatchSource = TextFormatMinibatchSource(L"Train.ctf", { { L"features", inputDim, true, L"x" }, { L"labels", numOutputClasses, false, L"y" } }, 0);
    auto featureStreamInfo = minibatchSource->StreamInfo(features);
    auto labelStreamInfo = minibatchSource->StreamInfo(labels);

    const size_t minibatchSize = 200;
    auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
    auto actualMBSize = minibatchData[labelStreamInfo].m_numSamples;

    LearningRatesPerSample learningRateSchedule({ { 20000, 0.0005 }, { 2, 0.00025 } }, actualMBSize);

    Internal::ResetUniqueId();

    auto classifierOutput = LSTMSequenceClassiferNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");
    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");
    auto prediction = CNTK::ClassificationError(classifierOutput, labels, L"classificationError");
    auto learner = SGDLearner(classifierOutput->Parameters(), learningRateSchedule);
    Trainer trainer(classifierOutput, trainingLoss, prediction, { learner }); // load all statics
    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);

    Internal::ResetUniqueId();
    auto classifierOutput1 = LSTMSequenceClassiferNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput1");
    auto trainingLoss1 = CNTK::CrossEntropyWithSoftmax(classifierOutput1, labels, L"lossFunction1");
    auto prediction1 = CNTK::ClassificationError(classifierOutput1, labels, L"classificationError1");
    auto learner01 = SGDLearner(classifierOutput1->Parameters(), learningRateSchedule);
    Trainer trainer1(classifierOutput1, trainingLoss1, prediction1, { learner01 });
    
    Internal::ResetUniqueId();
    auto classifierOutput2 = LSTMSequenceClassiferNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput2");
    auto trainingLoss2 = CNTK::CrossEntropyWithSoftmax(classifierOutput2, labels, L"lossFunction2");
    auto prediction2 = CNTK::ClassificationError(classifierOutput2, labels, L"classificationError2");
    auto learner02 = SGDLearner(classifierOutput2->Parameters(), learningRateSchedule);
    Trainer trainer2(classifierOutput2, trainingLoss2, prediction2, { learner02 });
   
    //1) this works!
    trainer1.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    trainer2.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);

    /*trainer1.SaveCheckpoint(L"trainer.checkpoint.v2_3", false);
    trainer1.SaveCheckpoint(L"trainer.checkpoint.v1_31", true);
    trainer2.RestoreFromCheckpoint(L"trainer.checkpoint.v2_3", false);
    trainer2.SaveCheckpoint(L"trainer.checkpoint.v1_32", true);*/

    //trainer1.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    //trainer1.SaveCheckpoint(L"trainer.checkpoint.v1_1", false);
    //trainer2.SaveCheckpoint(L"trainer.checkpoint.v1_2", true);

    //trainer1.RestoreFromCheckpoint(L"trainer.checkpoint.v1");
    
    //trainer2.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    //trainer2.SaveCheckpoint(L"trainer.checkpoint.v2", false);
    
    trainer1.RestoreFromCheckpoint(L"trainer.checkpoint.v1_1", false);
    trainer2.RestoreFromCheckpoint(L"trainer.checkpoint.v1_2", true);


    
    trainer1.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    trainer2.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
       

    //check learners!

   // trainer1.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    //trainer2.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);

   /* assert(trainer1.ParameterLearners().size() == 1 && trainer2.ParameterLearners().size() == 1);


    auto& learner1 = trainer1.ParameterLearners()[0];
   auto& learner2 = trainer2.ParameterLearners()[0];

    assert(learner1->Parameters().size() == learner2->Parameters().size());

    std::unordered_map<std::wstring, Variable> map1;
    for (auto& p : learner1->Parameters())
    {
        map1[p.Uid()] = p;
    }
    std::unordered_map<std::wstring, Variable> map2;
    for (auto& p : learner2->Parameters())
    {
        map2[p.Uid()] = p;
    }

    assert(map1.size() == map2.size());

    for (auto& it : map1)
    {
        auto& p1 = reinterpret_cast<Parameter&>(it.second);
        auto& p2 = reinterpret_cast<Parameter&>(map2[it.first]);
        assert(p1.Uid() == p2.Uid());
        if ( !AreEqual<float>(*(p1.Value().get()), *(p2.Value().get())))
        {
            assert(false);
        }
        if ( !AreEqual<float>(*(learner1->GradinetXXX(p1).get()), *(learner2->GradinetXXX(p2).get())))
        {
            assert(false);
        }
    }*/

    


    Internal::ResetUniqueId();
    int k = 10;
    while (k-- > 0)
    {
        auto f1 = trainer1.XXX();
        auto f2 = trainer2.XXX();

        auto parameters1 = f1->Inputs();
        auto parameters2 = f2->Inputs();

        assert(parameters1.size() == parameters2.size());

        for (int i = 0; i < parameters1.size(); ++i)
        {
            auto uid1 = parameters1[i].Uid();
            auto uid2 = parameters2[i].Uid();

            if (uid1 != uid2)
            {
                assert(false);
            }

            if (parameters1[i].IsConstant())
            {
                assert(parameters2[i].IsConstant());
                auto c1 = Constant(parameters1[i]);
                auto c2 = Constant(parameters2[i]);

                if (!AreEqual<float>(*(c1.Value().get()), *(c2.Value().get())))
                {
                    assert(false);
                }
            } 
            else if (parameters1[i].IsParameter())
            {
                assert(parameters2[i].IsParameter());
                auto p1 = Parameter(parameters1[i]);
                auto p2 = Parameter(parameters2[i]);

               

                /*if ( !AreEqual<float>(*(p1.Value().get()), *(p2.Value().get())))
                {
                    assert(false);
                }*/
            }
        
        }


     
        double mbLoss1 = trainer1.PreviousMinibatchLossAverage();
        double mbLoss2 = trainer2.PreviousMinibatchLossAverage();
        if (mbLoss1 != mbLoss2)
            throw std::runtime_error("Post checkpoint restoration training loss does not match expectation");
        
    }

   


    //for (int i = 0; i < 3; ++i)
    //{
    //    
    //   

    //   /* Dictionary model = Function::Save(classifierOutput);
    //    Dictionary checkpoint = learner->Serialize();

    //    auto classifierOutputReloaded = Function::Load(model, device);

    //    std::unordered_map<Variable, Variable> replacements;
    //    const auto& inputs = classifierOutputReloaded->Inputs();
    //    for (const auto& input : inputs)
    //    {
    //        if (input.IsPlaceholder() && input.Uid() == features.Uid())
    //        {
    //            replacements[input] = features;
    //        }
    //    }

    //    classifierOutputReloaded->ReplacePlaceholders(replacements);

    //    auto trainingLossReloaded = CNTK::CrossEntropyWithSoftmax(classifierOutputReloaded, labels, L"lossFunction");
    //    auto predictionReloaded = CNTK::ClassificationError(classifierOutputReloaded, labels, L"classificationError");
    //    auto learnerReloaded = SGDLearner(classifierOutputReloaded->Parameters(), learningRateSchedule);
    //    learnerReloaded->RestoreFromCheckpoint(checkpoint);


    //    auto parameters1 = classifierOutput->Parameters();
    //    auto parameters2 = classifierOutputReloaded->Parameters();

    //    assert(parameters1.size() == parameters2.size());

    //    for (int i = 0; i < parameters1.size(); ++i)
    //    {
    //        if (!AreEqual<float>(*(parameters1[i].Value().get()), *(parameters2[i].Value().get())))
    //        {
    //            return;
    //        }
    //    }*/

    //    //Trainer trainerReloaded(classifierOutputReloaded, trainingLossReloaded, predictionReloaded, { learnerReloaded });


    //    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    //    //trainerReloaded.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);

    //    double mbLoss1 = trainer.PreviousMinibatchLossAverage();

    //    trainer.RestoreFromCheckpoint(L"trainer.checkpoint.v2", true);
    //        
    //    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);


    //    double mbLoss2 = trainer.PreviousMinibatchLossAverage();

    //    if (mbLoss1 != mbLoss2)
    //        throw std::runtime_error("Post checkpoint restoration training loss does not match expectation");
    //    
    //}
}

void TestLegacyModelSaving(const DeviceDescriptor& device)
{
    const size_t inputDim = 2000;
    const size_t cellDim = 25;
    const size_t hiddenDim = 25;
    const size_t embeddingDim = 50;
    const size_t numOutputClasses = 5;

    auto features = InputVariable({ inputDim }, true /*isSparse*/, DataType::Float, L"features");
    

    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels", { Axis::DefaultBatchAxis() });
    

    auto minibatchSource = TextFormatMinibatchSource(L"Train.ctf", { { L"features", inputDim, true, L"x" }, { L"labels", numOutputClasses, false, L"y" } }, 0);
    auto featureStreamInfo = minibatchSource->StreamInfo(features);
    auto labelStreamInfo = minibatchSource->StreamInfo(labels);

    const size_t minibatchSize = 200;
    auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
    auto actualMBSize = minibatchData[labelStreamInfo].m_numSamples;

    LearningRatesPerSample learningRateSchedule({ { 2, 0.0005 }, { 2, 0.00025 } }, actualMBSize);


    Internal::ResetUniqueId();
    auto classifierOutput = LSTMSequenceClassiferNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");
    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");
    auto prediction = CNTK::ClassificationError(classifierOutput, labels, L"classificationError");
    auto learner = SGDLearner(classifierOutput->Parameters(), learningRateSchedule);
    Trainer trainer(classifierOutput, trainingLoss, prediction, { learner });

    const wchar_t* modelFile = L"seq2seq.model";

    trainer.RestoreFromCheckpoint(modelFile);

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);

    auto MB2Loss = trainer.PreviousMinibatchLossAverage();
   
    /*Dictionary checkpoint = learner->Serialize();
    SaveAsLegacyModel(classifierOutput, modelFile);

    learner->RestoreFromCheckpoint(checkpoint);*/ 

     trainer.SaveCheckpoint(modelFile);

      trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
//    auto MB2Loss = trainer.PreviousMinibatchLossAverage();

    /*trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    auto MB2Loss = trainer.PreviousMinibatchLossAverage();
    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);

    classifierOutput->RestoreFromLegacyModel(modelFile);

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    auto postRestoreMB2Loss = trainer.PreviousMinibatchLossAverage();
    FloatingPointCompare(postRestoreMB2Loss, MB2Loss, "Post checkpoint restoration training loss does not match expectation");*/

    Internal::ResetUniqueId();
    auto classifierOutput2 = LSTMSequenceClassiferNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");
        //LoadLegacyModel(DataType::Float, modelFile, DeviceDescriptor::CPUDevice());
        //LSTMSequenceClassiferNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");
    auto trainingLoss2 = CNTK::CrossEntropyWithSoftmax(classifierOutput2, labels, L"lossFunction");
    auto prediction2 = CNTK::ClassificationError(classifierOutput2, labels, L"classificationError");
    auto learner2 = SGDLearner(classifierOutput2->Parameters(), learningRateSchedule);
   
    Trainer trainer2(classifierOutput2, trainingLoss2, prediction2, { learner2 });


    trainer2.RestoreFromCheckpoint(modelFile);

    //classifierOutput2->RestoreFromLegacyModel(modelFile);
    //learner2->RestoreFromCheckpoint(checkpoint);

    


    trainer2.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    auto postRestoreMB2Loss = trainer2.PreviousMinibatchLossAverage();

   /* SaveAsLegacyModel(classifierOutput, modelFile);

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);

    classifierOutput->RestoreFromLegacyModel(modelFile);

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    postRestoreMB2Loss = trainer.PreviousMinibatchLossAverage();*/
    FloatingPointCompare(postRestoreMB2Loss, MB2Loss, "Post checkpoint restoration training loss does not match expectation");
}


void TestSerialization(const DeviceDescriptor& device)
{
    const size_t inputDim = 2000;
    auto inputVar = InputVariable({ inputDim }, true /*isSparse*/, DataType::Float, L"input_variable");

    TestFunctionSaveAndLoad(FullyConnectedLinearLayer(inputVar, 70, device), device);

    
    auto nonLinearity = std::bind(Sigmoid, std::placeholders::_1, L"");
    auto net = FullyConnectedDNNLayer(inputVar, 40, device, nonLinearity);
    for (size_t i = 1; i < 3; ++i)
        net = FullyConnectedDNNLayer(inputVar, 40, device, nonLinearity);

    TestFunctionSaveAndLoad(net, device);

    const size_t cellDim = 25;
    const size_t hiddenDim = 25;
    const size_t embeddingDim = 50;
    const size_t numOutputClasses = 5;

    auto classifier = LSTMSequenceClassiferNet(inputVar, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");

     TestFunctionSaveAndLoad(classifier, device);
}

void SerializationTests()
{

   //ComparisonTests(DeviceDescriptor::CPUDevice());



   /* TestDictionarySerialization(4);
    TestDictionarySerialization(8);
    TestDictionarySerialization(16);

    TestLearnerSerialization<float>(5, DeviceDescriptor::CPUDevice());
    TestLearnerSerialization<double>(10, DeviceDescriptor::CPUDevice());
*/
    TestLegacyModelSaving(DeviceDescriptor::CPUDevice());

//    TestSerialization(DeviceDescriptor::CPUDevice());

    //TestFunctionSerialization(DeviceDescriptor::CPUDevice());
    TestModelSerialization(DeviceDescriptor::CPUDevice());

#ifndef CPUONLY
    TestLearnerSerialization<float>(5, DeviceDescriptor::GPUDevice(0));
    TestLearnerSerialization<double>(10, DeviceDescriptor::GPUDevice(0));;
    TestModelSerialization(DeviceDescriptor::GPUDevice(0));
    TestLegacyModelSaving(DeviceDescriptor::GPUDevice(0));
#endif
 
}
