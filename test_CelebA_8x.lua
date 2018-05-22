require 'paths'
require 'image'
require 'math'
require 'nn'
require 'nngraph'
require 'cudnn'
require 'cunn'
require 'cutorch'
require 'torch'
local matio = require 'matio'

function byte2float(src)
    local conversion = false
    local dest = src
    if src:type() == "torch.ByteTensor" then
        conversion = true
	dest = src:float():div(255.0)
    end
    return dest, conversion
end

local function rgb2y_matlab(x)
    local y = torch.Tensor(1, x:size(2), x:size(3)):zero()
    x = byte2float(x)
    y:add(x[1] * 65.481)
    y:add(x[2] * 128.553)
    y:add(x[3] * 24.966)
    y:add(16.0)
    return y:byte():float()
end

local function YMSE(x1, x2)
    local x1_2 = rgb2y_matlab(x1)
    local x2_2 = rgb2y_matlab(x2)
    return (x1_2 - x2_2):pow(2):mean()
end

local function PSNR(x1, x2)
    local mse = math.max(YMSE(x1, x2), 1)
    return 10 * math.log10((255.0 * 255.0) / mse)
end

scale = 8
type_ = 'mse'
if type_ == 'mse' then
    -- celebA_20W_MSE_8x
    model = torch.load('./models/CelebA_8x.t7')  
elseif type_ == 'gan' then
    -- celebA_20W_GAN_8x
    model = torch.load('./models/CelebA_GAN_8x.t7')    
end

--------------------------------------------------------------------------------
-- Main loop
--------------------------------------------------------------------------------
path_LR  = 'celeba_lr'
path_HR  = 'celeba_hr'
dataset_ = 'celeba'
file     = io.open('./data/fileList_' .. path_LR .. '.txt', 'r')
count__  = 0
for line in file:lines() do
    count__ = count__+1
end

RMSE     = 0
counti   = 1
hei_     = 128
wid_     = 128
nStack_  = 2
file_num = count__
psnr_set = torch.Tensor(file_num)
file     = io.open('./data/fileList_' .. path_LR .. '.txt', 'r')
file_HR  = io.open('./data/fileList_' .. path_HR .. '.txt', 'r')

for line in file:lines() do
    print(counti)
    path_input      = unpack(line:split(" "))
    line_hr         = file_HR:read('*line')
    path_input_inv  = string.reverse(path_input)
    pos_            = string.find(path_input_inv, '/')
    len_            = string.len(path_input)
    filename_       = string.sub(path_input, len_-pos_+2,len_)
    path_label      = unpack(line_hr:split(" "))

    inp             = image.load(path_input)
    label_          = image.load(path_label)

    local output    = model:forward(inp:view(1,3,hei_,wid_):cuda())
    image_save      = output[nStack_*2][1]
    frnet_psnr      = PSNR(image_save:double(),label_:double())
    psnr_set[counti] = frnet_psnr
    counti           = counti + 1


    if type_ == 'mse' then
        image.save('./results/celeba_fsrnet_8x/output_2_'.. filename_,image_save)
    elseif type_ == 'gan' then
        image.save('./results/celeba_fsrgan_8x/output_2_'.. filename_,image_save)
    end

    collectgarbage()
end
avg_psnr = psnr_set:mean()
if type_ == 'mse' then
    print('************ fsrnet_avg: ' .. avg_psnr .. ' **************')
elseif type_ == 'gan' then
    print('************ fsrgan_avg: ' .. avg_psnr .. ' **************')
end

file:close()
file_HR:close()
