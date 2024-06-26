import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    uncond_prompt=None,  # negative prompt
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        if idle_device:
            to_idle: lambda x:x.to(idle_device)
        else:
            to_idle: lambda x:x
        
        generator=  torch.Generator(device=device) #This will generate the random numbers for noise

        if seed is None:
            generate.seed()
        else:
            generate.manual_seed(seed)
        
        clip=models["clip"]
        clip.to(device)

        if do_cfg:
            # convert the prompt into using tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt],padding='max_length',max_length=77).input_ids
            # (Batch_size, Seq_len)
            cond_tokens  =torch.tensor(cond_tokens ,dtype=torch.long,device=device)
            # (Batch_Size,Seq_len) ->(Batch_Size,Seq_len,Dim)
            cond_context= clip(cond_tokens )

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt],padding='max_length',max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens ,dtype=torch.long,device=device)
            # (Batch_Size,Seq_len) ->(Batch_Size,Seq_len,Dim)
            uncond_context= clip(uncond_tokens)

            #(2, Seq_len,Dim) = (2, 77, 768)

            context=torch.cat([cond_tokens,uncond_context])
        else:
            #convert it into lis of tokens
            tokens = tokenizer.batch_encode_plus([prompt],padding='max_length',max_length=77).input_ids
            tokens  =torch.tensor(cond_tokens ,dtype=torch.long,device=device)
            context= clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler {sampler_name}")
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder=models['encoder']
            encoder.to(device)
            

            input_image_tensor=input_image.resize((WIDTH,HEIGHT))

            input_image_tensor= np.array(input_image_tensor)

            #(Height, width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor,dtype=torch.float32)

            input_image_tensor = rescale(input_image_tensor,(0,255),(-1,1))
            #(Height,width, channel) -> (Batch_size,Height,width, channel)
            input_image_tensor=input_image_tensor.unsqueeze(0)
            #(Batch_size,Height,width, channel) -> (Batch_size,channel,Height,width)
            input_image_tensor=input_image_tensor.permute(0,3,1,2)

            encoder_noise= torch.randn(latents_shape,generator=generator,device=device)
            #run the image through the encoder of the VAE

            latents= encoder(input_image_tensor,encoder_noise)

            sampler.set_strength(strength=strength)
            latents= sampler.add_noise(latents,sampler.timesteps[0]) # if we set timesteps =1 it is the maxmimum noise

            to_idle(encoder)
        else:
            # if we are doing text-to image ,start random noise N(0,I)
            latents= torch.randn(latents_shape,generate=generate,device=device)

            diffusion =models['diffusion']
            diffusion.to(device)

            timesteps =tqdm(sampler.timesteps)
            for i, timesteps in enumerate(timesteps):

                #(1,320)
                time_embedding=get_time_embeddings(timesteps).to(device)

                #(Batch_size,4,latents_height,latents_width)

                model_input = latents
                if do_cfg:

                    #(Batch_size,4,latents_height,latents_width)->(2*Batch_size,4,latents_height,latents_width)
                    model_input=model_input.repeat(2,1,1,1)

                    # model output is the predictor noise by the UNET
                    model_input = diffusion(model_input,context,time_embedding)


                    if do_cfg:
                        output_cond,output_uncond= model_output.chunk(2)
                        model_output=cfg_scale * (output_cond- output_uncond) + output_uncond
                    























































             




        

















