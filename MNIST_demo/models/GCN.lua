require 'nn'
local utils = paths.dofile '../utils.lua'

return function (opt)
    assert(opt.orientation, 'missing orientation')
    opt.channel=opt.orientation

    local model = nn.Sequential()

    model:add(nn.View(-1, 32, 32))
    model:add(nn.Replicate(opt.channel, 2))

    model:add(nn.GCConv(1, 10, opt.orientation, 1,5,5,1,1,2,2):noBias())
    model:add(nn.SpatialBatchNormalization(10*opt.channel,1e-3))
    model:add(nn.ReLU())             

    model:add(nn.GCConv(10, 20, opt.orientation, 2,5,5):noBias())
    model:add(nn.SpatialBatchNormalization(20*opt.channel,1e-3))
    model:add(nn.ReLU())             
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) 
    
    
    model:add(nn.GCConv(20, 40, opt.orientation, 3,5,5):noBias())
    model:add(nn.SpatialBatchNormalization(40*opt.channel,1e-3))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    

    model:add(nn.GCConv(40, 80, opt.orientation, 4,5,5):noBias())
    model:add(nn.SpatialBatchNormalization(80*opt.channel,1e-3))
    model:add(nn.ReLU())      


    model:add(nn.View(80, opt.channel))
    model:add(nn.Max(2, 2))
    nFeatureDim =80

    model:add(nn.View(nFeatureDim))
    model:add(nn.Linear(nFeatureDim, 1024))
    model:add(nn.ReLU())    
    model:add(nn.Dropout(0.5))
    model:add(nn.Linear(1024, 10))     
    
    return model
end
