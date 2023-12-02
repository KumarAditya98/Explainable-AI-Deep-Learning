#%%

import torch.nn as nn
import torch
import streamlit as st
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import os
import shutil
import zipfile
import io
from DataLoader import create_dataloader


st.title("GANs")

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
WORKERS = 8
SEED = 123
BETA1 = 0.5

uploaded_file = st.file_uploader("Choose a zip file", type="zip")

if 'phase1' not in st.session_state:
    st.session_state["phase1"] = False

if 'phase2' not in st.session_state:
    st.session_state["phase2"] = False
    
if 'phase3' not in st.session_state:
    st.session_state["phase3"] = False

if 'phase4' not in st.session_state:
    st.session_state["phase4"] = False

if uploaded_file:

    zip_content = uploaded_file.read()

    zip_buffer = io.BytesIO(zip_content)

    with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
        zip_ref.extractall("extracted_folder")

    os.chdir("./extracted_folder")
    shutil.rmtree("./__MACOSX")
    os.chdir("/home/ubuntu/FinalProject")

    with st.form("Data_Param"):
        
        st.write("Image Size:")
        IMAGE_SIZE = st.text_input("", key = "image_size")
        st.write("Channels:")
        CHANNELS = st.text_input("", key = "channels")
        st.write("Batch Size:")
        BATCH = st.text_input("", key = "batch_size")
        
        submit = st.form_submit_button("Submit")
        if submit:
            st.session_state["phase1"] = True
        
if st.session_state["phase1"] == True:
    
    IMAGE_SIZE = int(IMAGE_SIZE)
    CHANNELS = int(CHANNELS)
    BATCH = int(BATCH)
    GENERATOR = torch.Generator(device="cpu")
    GENERATOR.manual_seed(SEED)
    
    dataloader = create_dataloader(dataroot= "/home/ubuntu/FinalProject/extracted_folder/",
                                   IMAGE_SIZE=IMAGE_SIZE,
                                   BATCH=BATCH,
                                   WORKERS=WORKERS,
                                   GENERATOR=GENERATOR)

    real_batch = next(iter(dataloader))
    fig, ax = plt.subplots(figsize=(8,8))
    ax.axis("off")
    ax.set_title("Training Images")
    ax.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(),(1,2,0)))

    st.pyplot(fig)
    
    st.session_state["phase2"] = True
    
if st.session_state["phase2"] == True:
    with st.form("Model_Param"):
        
        st.write("Learning Rate:")
        LR = st.text_input("", key = "lr")
        st.write("Epochs:")
        EPOCHS = st.text_input("", key = "epochs")
        st.write("Size Generator Feature Maps:")
        GEN_FM = st.text_input("", key = "G_FM")
        st.write("Size Discriminator Feature Maps:")
        DIS_FM = st.text_input("", key = "D_FM")
        st.write("Latent Vector Size:")
        LATENT_VEC_SIZE = st.text_input("", key = "latent")
        
        submit = st.form_submit_button("Submit")
        if submit:
            st.session_state["phase3"] = True

if st.session_state["phase3"] == True:
    
    LR = float(LR)
    EPOCHS = int(EPOCHS)
    GEN_FM = int(GEN_FM)
    DIS_FM = int(DIS_FM)
    LATENT_VEC_SIZE = int(LATENT_VEC_SIZE)
    
    model = st.selectbox(
    'What model do you wish to choose?',
    ('Select model', 'dcgan', 'gan2', 'gan3'))
    
    if model == "Select model":
        st.stop()

    if model == "dcgan":
        
        class Generator(nn.Module):
            def __init__(self):
                super(Generator, self).__init__()
                self.main = nn.Sequential(
                    # input is Z, going into a convolution
                    nn.ConvTranspose2d(LATENT_VEC_SIZE, GEN_FM * 8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(GEN_FM * 8),
                    nn.ReLU(),
                    # state size. ``(GEN_FM*8) x 4 x 4``
                    nn.ConvTranspose2d(GEN_FM * 8, GEN_FM * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(GEN_FM * 4),
                    nn.ReLU(),
                    # state size. ``(GEN_FM*4) x 8 x 8``
                    nn.ConvTranspose2d( GEN_FM * 4, GEN_FM * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(GEN_FM * 2),
                    nn.ReLU(),
                    # state size. ``(GEN_FM*2) x 16 x 16``
                    nn.ConvTranspose2d( GEN_FM * 2, GEN_FM, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(GEN_FM),
                    nn.ReLU(),
                    # state size. ``(GEN_FM) x 32 x 32``
                    nn.ConvTranspose2d( GEN_FM, CHANNELS, 4, 2, 1, bias=False),
                    nn.Tanh()
                    # state size. ``(nc) x 64 x 64``
                )

            def forward(self, input):
                return self.main(input)

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
                # input is ``(nc) x 64 x 64``
                nn.Conv2d(CHANNELS, DIS_FM, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(DIS_FM) x 32 x 32``
                nn.Conv2d(DIS_FM, DIS_FM * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(DIS_FM * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(DIS_FM*2) x 16 x 16``
                nn.Conv2d(DIS_FM * 2, DIS_FM * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(DIS_FM * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(DIS_FM*4) x 8 x 8``
                nn.Conv2d(DIS_FM * 4, DIS_FM * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(DIS_FM * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(DIS_FM*8) x 4 x 4``
                nn.Conv2d(DIS_FM * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)

    netD = Discriminator().to(device)
    netG = Generator().to(device)
    
    st.session_state["phase4"] = True

if st.session_state["phase4"] == True:
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

    fixed_noise = torch.randn(64, LATENT_VEC_SIZE, 1, 1, device=device)

    real_label = 1.
    fake_label = 0.
    
    last_f1, last_f2, last_f3, last_f4, last_f5 = None, None, None, None, None

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(EPOCHS):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
        
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, LATENT_VEC_SIZE, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            
            # Output training stats
            if i % 50 == 0:
                st.markdown('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, EPOCHS, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            iters += 1
        
        real_batch = next(iter(dataloader))
        
        # Plot the real images
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(8,8))
        ax1.axis("off")
        ax1.set_title("Real Images")
        ax1.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

        # Plot the fake images from the last epoch
        ax2.axis("off")
        ax2.set_title("Fake Images")
        ax2.imshow(np.transpose(img_list[-1],(1,2,0)))
        
        st.pyplot(fig)
        
        img_list = []
    
    # fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(8,8))
    # ax1.axis("off")
    # ax1.set_title("Real Image")
    # ax1.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:1], padding=5, normalize=True).cpu(),(1,2,0)))
    
    # # ax2.axis("off")
    # # ax2.set_title("Feature Map 1")
    # # ax2.imshow(np.transpose(f1[-1],(1,2,0)))
    # st.pyplot(fig)

    # plt.figure(figsize=(10,5))
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(G_losses,label="G")
    # plt.plot(D_losses,label="D")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()






    
    
    
#%%