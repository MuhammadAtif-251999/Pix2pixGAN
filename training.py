from tensorflow.keras.optimizers import Adam

def define_gan(g_model, d_model, image_shape):
    d_model.trainable = False  # Freeze Discriminator during Generator training
    
    in_src = Input(shape=image_shape)
    gen_out = g_model(in_src)  # Generator output
    dis_out = d_model([in_src, gen_out])  # Discriminator output
    
    model = Model(in_src, [dis_out, gen_out])  
    opt = Adam(beta_1=0.9, beta_2=0.999, learning_rate=0.0001)
    
    # Compile with custom loss function
    model.compile(loss=['binary_crossentropy', custom_loss],  
                  optimizer=opt,  
                  loss_weights=[1, 100])  # Discriminator loss (BCE) and Generator loss (Custom)
    
    return model

def generate_real_samples(dataset, n_samples, patch_shape):
    trainA, trainB = dataset
    ix = randint(0, trainA.shape[0], n_samples)
    X1, X2 = trainA[ix], trainB[ix]
    y = ones((n_samples, patch_shape, 1))
    return [X1, X2], y

def generate_fake_samples(g_model, samples, patch_shape):
    X = g_model.predict(samples)
    y = zeros((len(X), patch_shape, 1))
    return X,y

image_shape=src_images.shape[1:]

d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)



def summerize_performance(step, g_model, dataset, n_samples=3):
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_realA[i])
        
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(X_fakeB[i])
    
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples*2 + i)
        plt.axis('off')
        plt.imshow(X_realB[i])
        
    filename1 = 'plot_%06d.png' % (step+1)
    plt.savefig(filename1)
    plt.close()
    
    filename2 = 'model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))
    
def train(d_model, g_model, gan_model, dataset, n_epochs = 30, n_batch = 10):
    n_patch = d_model.output_shape[1]
    trainA, trainB = dataset
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs
    
    for i in range(n_steps):
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [ y_real, X_realB])
        print( '>%d, d1[%.3f] d2[%.3f]  g[%.3f]'  %  (i+1, d_loss1, d_loss2, g_loss))
        
        if (i+1) % (bat_per_epo * 10 ) == 0:
            summerize_performance(i, g_model, dataset)
