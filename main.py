import config
import dataset
import model
import training

def run():
    # 1. Load Data
    (train_src, train_gt), _ = dataset.get_datasets()
    
    # 2. Define Models
    img_shape = (config.HEIGHT, config.WIDTH, config.CHANNELS)
    g_model = model.define_generator(img_shape)
    d_model = model.define_discriminator(img_shape)
    
    # 3. Compile GAN
    gan_model = training.compile_gan(g_model, d_model, img_shape)
    
    # 4. Train
    print("Starting Training...")
    # Using your simple fit logic or a custom loop
    g_model.fit(train_src, train_gt, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS)
    
    # 5. Save Model
    g_model.save('generator_model.h5')

if __name__ == "__main__":
    run()