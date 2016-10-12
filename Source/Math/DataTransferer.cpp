//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "DataTransferer.h"
#include "GPUDataTransferer.h"

// Please do not remove, this is required to properly export some symbols
// of the exteranal symbols from the dll.
#include "cudalattice.h"

namespace
{
    using namespace msra::cuda;
    // Please do not remove, this is required to properly export some symbols
    // of the exteranal symbols from the dll.
    // More info :  https://blogs.msdn.microsoft.com/oldnewthing/20140321-00/?p=1433
    void NeededForExportingNeverCalled()
    {
        newlatticefunctions(0);
        newushortvector(0);
        newuintvector(0);
        newfloatvector(0);
        newdoublevector(0);
        newsizetvector(0);
        newlatticefunctions(0);
        newlrhmmdefvector(0);
        newlr3transPvector(0);
        newnodeinfovector(0);
        newedgeinfovector(0);
        newaligninfovector(0);
    }
}

namespace Microsoft { namespace MSR { namespace CNTK {

    DataTransfererPtr CreatePrefetchDataTransferer(int deviceId)
    {
        return std::make_shared<PrefetchGPUDataTransferer>(deviceId);
    }

} } }
