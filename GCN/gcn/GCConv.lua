local THNN = require 'nn.THNN'
local GCConv, parent = torch.class('nn.GCConv', 'nn.Module')

function GCConv:__init(nInputPlane, nOutputPlane, nOrientation, nScale, kW, kH, dW, dH, padW, padH)
   parent.__init(self)

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.nOrientation = nOrientation or 4
   self.nScale=nScale or 1
   self.nChannel=self.nOrientation

   self.kW = kW or 3
   self.kH = kH or 3
   
   self.gaborFilterBank = self:getGaborFilterBank(self.nOrientation, self.nScale, self.kH, self.kW) 
   self.dW = dW or 1
   self.dH = dH or 1
   
   self.padW = padW or 0
   self.padH = padH or self.padW

   self.weight = torch.rand(nOutputPlane, nInputPlane, self.nChannel, self.kH, self.kW)
   self.gradWeight = self.weight:clone():zero()

   self.shareWeight = torch.Tensor(nOutputPlane * self.nChannel, nInputPlane * self.nChannel * self.kH * self.kW)
   self.gradShareWeight = self.shareWeight:clone():zero()

   self.bias = torch.Tensor(nOutputPlane * self.nChannel)
   self.gradBias = self.bias:clone():zero()
 
   self:reset()
end

function GCConv:getGaborFilterBank(u,v,h,w)
	
	local Kmax = math.pi/2;
	local f=math.sqrt(2);
	local sigma=math.pi;
	local sqsigma=sigma^2;
    local postmean = math.exp(-sqsigma/2);
    local gfilter_real=torch.rand(u,h,w)
	for i=1,u do
	local theta=(i-1)/u*math.pi;

			local k=Kmax/f^(v-1);
                  xymax=-1e309;
			      xymin=1e309;
            for y=1,h do
				for x=1,w do
					local y1= y-((h+1)/2);
					local x1= x-((w+1)/2); 
					local tmp1=math.exp(-(k*k*(x1*x1+y1*y1)/(2*sqsigma))); 
					local tmp2=math.cos(k*math.cos(theta)*x1+k*math.sin(theta)*y1)-postmean;
					local tmp3=math.sin(k*math.cos(theta)*x1+k*math.sin(theta)*y1);

					gfilter_real[i][y][x]=k*k*tmp1*tmp2/sqsigma;
					
					
                    if gfilter_real[i][y][x]>xymax then
					   xymax=gfilter_real[i][y][x];
					end
					if gfilter_real[i][y][x]<xymin then
					   xymin=gfilter_real[i][y][x];
					end
                                   
				end

			end

            if h~=1 then
                for y=1,h do
                for x=1,h do
				gfilter_real[i][y][x]=(gfilter_real[i][y][x]-xymin)/(xymax-xymin);
                end
			    end
            end

	end
	
	return gfilter_real;
end

function GCConv:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function GCConv:reset(stdv)
   local n = self.kW*self.kH*self.nOutputPlane*self.nChannel
   self.weight:normal(0, math.sqrt(2/n))
   self.bias:zero()
end

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
     self._gradOutput = self._gradOutput or gradOutput.new()
     self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
     gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

function GCConv:updateOutput(input)

   input.gcn.GOF_Producing(
      self,
      self.weight,
      self.gaborFilterBank,
      self.shareWeight,
      self.kW, self.kH, self.nInputPlane, self.nOutputPlane, self.nChannel
   )

   self.finput = self.finput or input.new()
   self.fgradInput = self.fgradInput or input.new()
   if self.padding then
      self.padW = self.padding
      self.padH = self.padding
      self.padding = nil
   end
   input = makeContiguous(self, input)
   input.THNN.SpatialConvolutionMM_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.shareWeight:cdata(),
      THNN.optionalTensor(self.bias),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH
   )
   return self.output
end

function GCConv:updateGradInput(input, gradOutput)
   assert(input.THNN, torch.type(input)..'.THNN backend not imported')
   if self.gradInput then
      input, gradOutput = makeContiguous(self, input, gradOutput)
      input.THNN.SpatialConvolutionMM_updateGradInput(
         input:cdata(),
         gradOutput:cdata(),
         self.gradInput:cdata(),
         self.shareWeight:cdata(),
         self.finput:cdata(),
         self.fgradInput:cdata(),
         self.kW, self.kH,
         self.dW, self.dH,
         self.padW, self.padH
      )
      return self.gradInput
   end
end

function GCConv:accGradParameters(input, gradOutput, scale)
   assert(input.THNN, torch.type(input)..'.THNN backend not imported')
   scale = scale or 1
   input, gradOutput = makeContiguous(self, input, gradOutput)
   assert((self.bias and self.gradBias) or (self.bias == nil and self.gradBias == nil))
   
   self.gradShareWeight:zero()

   input.THNN.SpatialConvolutionMM_accGradParameters(
      input:cdata(),
      gradOutput:cdata(),
      self.gradShareWeight:cdata(),
      THNN.optionalTensor(self.gradBias),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH,
      scale
   )

   input.gcn.GOF_BPAlign(
      self,
      self.gradWeight,
      self.gradShareWeight,
      self.kW, self.kH, self.nInputPlane, self.nOutputPlane, self.nChannel, self.nOrientation, self.nScale, self.gaborFilterBank
   )
end

function GCConv:type(type,tensorCache)
   self.finput = self.finput and torch.Tensor()
   self.fgradInput = self.fgradInput and torch.Tensor()
   return parent.type(self,type,tensorCache)
end

function GCConv:__tostring__()
   local s = string.format('%s([%d] %d -> %d, %dx%d', torch.type(self),
         self.nChannel, self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   if self.bias then
      return s .. ')'
   else
      return s .. ') without bias'
   end
end

function GCConv:clearState()
   nn.utils.clear(self, 'finput', 'fgradInput', '_input', '_gradOutput')
   return parent.clearState(self)
end

