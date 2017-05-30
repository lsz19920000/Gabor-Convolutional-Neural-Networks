local cudnn = require 'cudnn'
local GCConv, parent = torch.class('cudnn.GCConv', 'nn.GCConv')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

local autotunerCache = {}
autotunerCache[1] = {} 
autotunerCache[2] = {} 
autotunerCache[3] = {} 

function GCConv:__init(nInputPlane, nOutputPlane, nOrientation, nScale, kW, kH, dW, dH, padW, padH, groups)
    local delayedReset = self.reset
    self.reset = function() end
    parent.__init(self, nInputPlane, nOutputPlane, nOrientation, nScale, kW, kH, dW, dH, padW, padH)
    self.reset = delayedReset
    self.groups = groups or 1
    self.nChannel=nOrientation
    assert((nInputPlane * nChannel) % self.groups == 0,
           'nInputPlane x nChannel should be divisible by nGroups')
    assert((nOutputPlane * nChannel) % self.groups == 0,
           'nOutputPlane x nChannel should be divisible by nGroups')
    self.shareWeight = torch.Tensor(nOutputPlane * self.nChannel, nInputPlane * self.nChannel / self.groups, self.kH, self.kW)
    self.gradShareWeight = torch.Tensor(nOutputPlane * self.nChannel, nInputPlane * self.nChannel / self.groups, self.kH, self.kW)
    self:reset()
    self.reset = nil
end

function GCConv:resetShareWeightDescriptors()
    assert(torch.typename(self.shareWeight) == 'torch.CudaTensor',
           'Only Cuda supported duh!')
    assert(torch.typename(self.bias) == 'torch.CudaTensor' or not self.bias,
           'Only Cuda supported duh!')
    self.groups = self.groups or 1
    self.shareWeightDesc = ffi.new('struct cudnnFilterStruct*[1]')
    errcheck('cudnnCreateFilterDescriptor', self.shareWeightDesc)
    local desc = torch.IntTensor({self.nOutputPlane*self.nChannel/self.groups,
                              self.nInputPlane*self.nChannel/self.groups,
                              self.kH, self.kW})
    errcheck('cudnnSetFilterNdDescriptor', self.shareWeightDesc[0],
             'CUDNN_DATA_FLOAT', 'CUDNN_TENSOR_NCHW', 4,
             desc:data());
    local function destroyWDesc(d)
        errcheck('cudnnDestroyFilterDescriptor', d[0]);
    end
    ffi.gc(self.shareWeightDesc, destroyWDesc)

    if self.bias then
        self.biasDesc = cudnn.toDescriptor(self.bias:view(1, self.nOutputPlane * self.nChannel,1,1))
    end
end

function GCConv:fastest(mode)
    if mode == nil then mode = true end
    self.fastest_mode = mode
    self.iSize = self.iSize or torch.LongStorage(4)
    self.iSize:fill(0)
    return self
end

function GCConv:setMode(fmode, bdmode, bwmode)
    if fmode ~= nil then
        self.fmode = fmode
    end
    if bdmode ~= nil then
        self.bdmode = bdmode
    end
    if bwmode ~= nil then
        self.bwmode = bwmode
    end
    self.iSize = self.iSize or torch.LongStorage(4)
    self.iSize:fill(0)
    return self
end

function GCConv:resetMode()
    self.fmode = nil
    self.bdmode = nil
    self.bwmode = nil
    return self
end

function GCConv:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function GCConv:createIODescriptors(input)
    local batch = true
    if input:dim() == 3 then
        input = input:view(1, input:size(1), input:size(2), input:size(3))
        batch = false
    end
    assert(input:dim() == 4 and input:isContiguous());
    self.iSize = self.iSize or torch.LongStorage(4):fill(0)
    if not self.iDesc or not self.oDesc or
        input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
    or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] then
        self.iSize = input:size()

        assert(self.nInputPlane * self.nChannel == input:size(2), 'input has to contain: '
                   .. self.nInputPlane * self.nChannel
                   .. ' feature maps, but received input of size: '
                   .. input:size(1) .. ' x ' .. input:size(2) ..
                   ' x ' .. input:size(3) .. ' x ' .. input:size(4))

        local input_slice = {{},{1,self.nInputPlane * self.nChannel/self.groups},{},{}}
        self.iDesc = cudnn.toDescriptor(input[input_slice])

        self.convDesc = ffi.new('struct cudnnConvolutionStruct*[1]')
        errcheck('cudnnCreateConvolutionDescriptor', self.convDesc)
        self.padH, self.padW = self.padH or 0, self.padW or 0
        local pad = torch.IntTensor({self.padH, self.padW})
        local stride = torch.IntTensor({self.dH, self.dW})
        local upscale = torch.IntTensor({1,1})
        errcheck('cudnnSetConvolutionNdDescriptor', self.convDesc[0],
                 2, pad:data(),
                 stride:data(), upscale:data(), 'CUDNN_CROSS_CORRELATION',
                 'CUDNN_DATA_FLOAT');
        local function destroyConvDesc(d)
            errcheck('cudnnDestroyConvolutionDescriptor', d[0]);
        end
        ffi.gc(self.convDesc, destroyConvDesc)

        local oSize = torch.IntTensor(4)
        local oSizeD = oSize:data()
        errcheck('cudnnGetConvolutionNdForwardOutputDim',
                 self.convDesc[0], self.iDesc[0],
                 self.shareWeightDesc[0], 4, oSizeD)
        oSize[2] = oSize[2] * self.groups
        self.output:resize(oSize:long():storage())

        local output_slice = {{},{1,self.nOutputPlane * self.nChannel/self.groups},{},{}}
        self.oDesc = cudnn.toDescriptor(self.output[output_slice])
        self.oDescForBias = cudnn.toDescriptor(self.output)

        -----------------------------------------------------------------------
        local function shape(x)
            local sz = x:size()
            local str = ''
            for i=1,sz:size() do
                str = str .. sz[i] .. 'x'
            end
            if #str > 0 then
                str = str:sub(1, #str-1)
            end
            return str
        end
        local autotunerHash = shape(self.shareWeight) .. ';'
            .. shape(input[input_slice]) .. ';'
            .. shape(self.output[output_slice])

        local maxBufSize = 0

        local algType = ffi.new("cudnnConvolutionFwdAlgo_t[?]", 1)
        local algSearchMode = 'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT'
        local algWorkspaceLimit = self.workspace_limit
            or (self.nInputPlane * self.nChannel * self.kH * self.kW * 4) -- 4 = sizeof int/float.

        if self.fastest_mode or cudnn.fastest == true then
            algSearchMode = 'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST'
        end

        if cudnn.benchmark then 
            if autotunerCache[1][autotunerHash] then
                algType[0] = autotunerCache[1][autotunerHash]
                if cudnn.verbose then
                   print('Autotuning SC FW: using cached algo = ', algType[0], ' for: ', autotunerHash)
                end
            else
                local perfResults = ffi.new("cudnnConvolutionFwdAlgoPerf_t[?]", 1)
                local intt = torch.IntTensor(1);
                errcheck('cudnnFindConvolutionForwardAlgorithm',
                         cudnn.getHandle(),
                         self.iDesc[0], self.shareWeightDesc[0],
                         self.convDesc[0], self.oDesc[0],
                         1, intt:data(), perfResults)
                algType[0] = perfResults[0].algo
                autotunerCache[1][autotunerHash] = perfResults[0].algo
                if cudnn.verbose then
                    print(string.format(
                              "\nAutotuning SC     Forward: Time: %3.5f Memory: %8d Algorithm: %d"
                                  .. " ShareWeight: %15s Input: %15s Output: %15s",
                              perfResults[0].time, tonumber(perfResults[0].memory),
                              tonumber(perfResults[0].algo),
                              shape(self.shareWeight), shape(input[input_slice]),
                              shape(self.output[output_slice])))
                end
            end
        else
            errcheck('cudnnGetConvolutionForwardAlgorithm',
                     cudnn.getHandle(),
                     self.iDesc[0], self.shareWeightDesc[0],
                     self.convDesc[0], self.oDesc[0],
                     algSearchMode, algWorkspaceLimit, algType)
        end
        algType[0] = self.fmode or algType[0]
        self.fwdAlgType = algType
        local bufSize = torch.LongTensor(1)
        errcheck('cudnnGetConvolutionForwardWorkspaceSize',
                 cudnn.getHandle(),
                 self.iDesc[0], self.shareWeightDesc[0],
                 self.convDesc[0], self.oDesc[0],
                 algType[0], bufSize:data())
        maxBufSize = math.max(maxBufSize, bufSize[1])

        local algType = ffi.new("cudnnConvolutionBwdFilterAlgo_t[?]", 1)
        local algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE'
        local algWorkspaceLimit = self.workspace_limit
            or (self.nInputPlane * self.nChannel * self.kH * self.kW * 4) -- 4 = sizeof int/float.
        if self.fastest_mode  or cudnn.fastest == true then
            algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST'
        end

        if cudnn.benchmark then 
            if autotunerCache[2][autotunerHash] then
                algType[0] = autotunerCache[2][autotunerHash]
                if cudnn.verbose then
                   print('Autotuning SC BW: using cached algo = ', algType[0], ' for: ', autotunerHash)
                end
            else
                local perfResults = ffi.new("cudnnConvolutionBwdFilterAlgoPerf_t[?]", 1)
                local intt = torch.IntTensor(1);
                errcheck('cudnnFindConvolutionBackwardFilterAlgorithm',
                         cudnn.getHandle(),
                         self.iDesc[0], self.oDesc[0],
                         self.convDesc[0], self.shareWeightDesc[0],
                         1, intt:data(), perfResults)
                algType[0] = perfResults[0].algo
                autotunerCache[2][autotunerHash] = perfResults[0].algo
                if cudnn.verbose then
                    print(string.format(
                              "Autotuning backwardFilter: Time: %3.5f Memory: %8d Algorithm: %d"
                                  .. " ShareWeight: %15s Input: %15s Output: %15s",
                              perfResults[0].time, tonumber(perfResults[0].memory),
                              tonumber(perfResults[0].algo),
                              shape(self.shareWeight), shape(input[input_slice]),
                              shape(self.output[output_slice])))
                end
            end
        else
            errcheck('cudnnGetConvolutionBackwardFilterAlgorithm',
                     cudnn.getHandle(),
                     self.iDesc[0], self.oDesc[0],
                     self.convDesc[0], self.shareWeightDesc[0],
                     algSearchMode, algWorkspaceLimit, algType)
        end
        algType[0] = self.bwmode or algType[0]
        self.bwdFilterAlgType = algType
        local bufSize = torch.LongTensor(1)
        errcheck('cudnnGetConvolutionBackwardFilterWorkspaceSize',
                 cudnn.getHandle(),
                 self.iDesc[0], self.oDesc[0],
                 self.convDesc[0], self.shareWeightDesc[0],
                 algType[0], bufSize:data())
        maxBufSize = math.max(maxBufSize, bufSize[1])

        local algType = ffi.new("cudnnConvolutionBwdDataAlgo_t[?]", 1)
        local algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE'
        local algWorkspaceLimit = self.workspace_limit
            or (self.nInputPlane * self.nChannel * self.kH * self.kW * 4) 
        if self.fastest_mode or cudnn.fastest == true then
            algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST'
        end
        if cudnn.benchmark then 
            if autotunerCache[3][autotunerHash] then
                algType[0] = autotunerCache[3][autotunerHash]
                if cudnn.verbose then
                   print('Autotuning SC BWD: using cached algo = ', algType[0], ' for: ', autotunerHash)
                end
            else
                local perfResults = ffi.new("cudnnConvolutionBwdDataAlgoPerf_t[?]", 1)
                local intt = torch.IntTensor(1);
                errcheck('cudnnFindConvolutionBackwardDataAlgorithm',
                         cudnn.getHandle(),
                         self.shareWeightDesc[0], self.oDesc[0],
                         self.convDesc[0], self.iDesc[0],
                         1, intt:data(), perfResults)
                algType[0] = perfResults[0].algo
                autotunerCache[3][autotunerHash] = perfResults[0].algo
                if cudnn.verbose then
                    print(string.format(
                              "Autotuning   backwardData: Time: %3.5f Memory: %8d Algorithm: %d"
                                  .. " ShareWeight: %15s Input: %15s Output: %15s\n",
                              perfResults[0].time, tonumber(perfResults[0].memory),
                              tonumber(perfResults[0].algo),
                              shape(self.shareWeight), shape(input[input_slice]),
                              shape(self.output[output_slice])))
                end
            end
        else
            errcheck('cudnnGetConvolutionBackwardDataAlgorithm',
                     cudnn.getHandle(),
                     self.shareWeightDesc[0], self.oDesc[0],
                     self.convDesc[0], self.iDesc[0],
                     algSearchMode, algWorkspaceLimit, algType)
        end
        algType[0] = self.bdmode or algType[0]
        self.bwdDataAlgType = algType
        local bufSize = torch.LongTensor(1)
        errcheck('cudnnGetConvolutionBackwardDataWorkspaceSize',
                 cudnn.getHandle(),
                 self.shareWeightDesc[0], self.oDesc[0],
                 self.convDesc[0], self.iDesc[0],
                 algType[0], bufSize:data())
        maxBufSize = math.max(maxBufSize, bufSize[1])

        self.extraBuffer = self.extraBuffer or cudnn.getSharedWorkspace()
        self.extraBufferSizeInBytes = self.extraBuffer:nElement() * 4 
        if maxBufSize > self.extraBufferSizeInBytes then
            self.extraBuffer:resize(math.ceil(maxBufSize/4))
            self.extraBufferSizeInBytes = maxBufSize
        end

        -----------------------------------------------------------------------
        local iH, iW = input:size(3), input:size(4)
        local kH, kW = self.kH, self.kW
        local oH, oW = oSize[3], oSize[4]
        self.input_offset = self.nInputPlane * self.nChannel / self.groups * iH * iW
        self.output_offset = self.nOutputPlane * self.nChannel / self.groups * oH * oW
        self.shareWeight_offset = self.nInputPlane * self.nChannel / self.groups * self.nOutputPlane * self.nChannel / self.groups * kH * kW

        if not batch then
            self.output = self.output:view(self.output:size(2),
                                           self.output:size(3),
                                           self.output:size(4))
        end
    end
end

local one = torch.FloatTensor({1});
local zero = torch.FloatTensor({0});

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:typeAs(input):resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput and not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   return input, gradOutput
end

function GCConv:updateOutput(input)
    if not self.shareWeightDesc then self:resetShareWeightDescriptors() end
    input = makeContiguous(self, input)
    self:createIODescriptors(input)

    input.gcn.GOF_Producing(
      self,
      self.weight,
      self.gaborFilterBank,
      self.shareWeight,
      self.kW, self.kH, self.nInputPlane, self.nOutputPlane, self.nChannel
    )

    for g = 0, self.groups - 1 do
        errcheck('cudnnConvolutionForward', cudnn.getHandle(),
                 one:data(),
                 self.iDesc[0], input:data() + g*self.input_offset,
                 self.shareWeightDesc[0], self.shareWeight:data() + g*self.shareWeight_offset,
                 self.convDesc[0], self.fwdAlgType[0],
                 self.extraBuffer:data(), self.extraBufferSizeInBytes,
                 zero:data(),
                 self.oDesc[0], self.output:data() + g*self.output_offset);
    end

    if self.bias then
        errcheck('cudnnAddTensor', cudnn.getHandle(),
                 one:data(), self.biasDesc[0], self.bias:data(),
                 one:data(), self.oDescForBias[0], self.output:data())
    end

    return self.output
end

function GCConv:updateGradInput(input, gradOutput)
    if not self.gradInput then return end
    self.gradInput:resizeAs(input)

    input, gradOutput = makeContiguous(self, input, gradOutput)
    assert(gradOutput:dim() == 3 or gradOutput:dim() == 4, 'gradOutput has to be 3D or 4D');
    if not self.shareWeightDesc then self:resetShareWeightDescriptors() end
    self:createIODescriptors(input)

    for g = 0,self.groups - 1 do
        errcheck('cudnnConvolutionBackwardData', cudnn.getHandle(),
                 one:data(),
                 self.shareWeightDesc[0], self.shareWeight:data() + g*self.shareWeight_offset,
                 self.oDesc[0], gradOutput:data() + g*self.output_offset,
                 self.convDesc[0],
                 self.bwdDataAlgType[0],
                 self.extraBuffer:data(), self.extraBufferSizeInBytes,
                 zero:data(),
                 self.iDesc[0], self.gradInput:data() + g*self.input_offset);
    end
    return self.gradInput
end

function GCConv:accGradParameters(input, gradOutput, scale)
    self.scaleT = self.scaleT or torch.FloatTensor(1):fill(1.0)
    self.scaleT = self.scaleT:float()
    scale = scale or 1.0
    self.scaleT[1] = scale

    input, gradOutput = makeContiguous(self, input, gradOutput)

    assert(gradOutput:dim() == 3 or gradOutput:dim() == 4, 'gradOutput has to be 3D or 4D');
    if not self.shareWeightDesc then self:resetShareWeightDescriptors() end
    self:createIODescriptors(input)

    if self.bias then
        errcheck('cudnnConvolutionBackwardBias', cudnn.getHandle(),
                 self.scaleT:data(),
                 self.oDescForBias[0], gradOutput:data(),
                 one:data(),
                 self.biasDesc[0], self.gradBias:data())
    end

    self.gradShareWeight:zero()

    for g = 0, self.groups - 1 do
        errcheck('cudnnConvolutionBackwardFilter', cudnn.getHandle(),
                 self.scaleT:data(),
                 self.iDesc[0], input:data() + g*self.input_offset,
                 self.oDesc[0], gradOutput:data() + g*self.output_offset,
                 self.convDesc[0],
                 self.bwdFilterAlgType[0],
                 self.extraBuffer:data(), self.extraBufferSizeInBytes,
                 one:data(),
                 self.shareWeightDesc[0], self.gradShareWeight:data() + g*self.shareWeight_offset);
    end

    input.gcn.GOF_BPAlign(
      self,
      self.gradWeight,
      self.gradShareWeight,
      self.kW, self.kH, self.nInputPlane, self.nOutputPlane, self.nChannel, self.nOrientation, self.nScale, self.gaborFilterBank
    )
end

function GCConv:clearDesc()
    self.shareWeightDesc = nil
    self.biasDesc = nil
    self.convDesc = nil
    self.iDesc = nil
    self.oDesc = nil
    self.oDescForBias = nil
    self.algType = nil
    self.fwdAlgType = nil
    self.bwdDataAlgType = nil
    self.bwdFilterAlgType = nil
    self.extraBuffer = nil
    self.extraBufferSizeInBytes = nil
    self.scaleT = nil
end

function GCConv:write(f)
    self:clearDesc()
    local var = {}
    for k,v in pairs(self) do
        var[k] = v
    end
    f:writeObject(var)
end

function GCConv:clearState()
   self:clearDesc()
   nn.utils.clear(self, '_input', '_gradOutput')
   return nn.Module.clearState(self)
end
