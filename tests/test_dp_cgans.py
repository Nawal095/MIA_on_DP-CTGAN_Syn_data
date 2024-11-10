import pandas as pd
from dp_cgans import DP_CGAN, __version__
from dp_cgans.__main__ import cli
from typer.testing import CliRunner

runner = CliRunner()

def test_dp_cgans():
    print(f'Testing DP_CGAN {__version__}')

    # Load your dataset (original)
    tabular_data = pd.read_csv("/Users/noshinnawal/Downloads/dp-ctgans/MelanomaLight.csv")

    model = DP_CGAN(
        epochs=10,  # number of training epochs
        batch_size=1000,  # size of each batch
        log_frequency=True,
        verbose=False,
        generator_dim=(128, 128, 128),
        discriminator_dim=(128, 128, 128),
        generator_lr=2e-4,
        discriminator_lr=2e-4,
        discriminator_steps=1,
        private=False,
        wandb=False
    )

    # Fit the model on your original data
    model.fit(tabular_data)

    # Generate synthetic samples
    sample = model.sample(100)

    # Test assertions
    assert len(sample) == 100
    print("Sample generation test passed.")

def test_cli():
    gen_size = 100
    result = runner.invoke(cli, [
        "gen", "/mnt/data/MelanomaLight.csv", 
        "--epochs", "2", 
        "--gen-size", str(gen_size)
    ])
    
    # Check if the CLI command executed successfully
    assert result.exit_code == 0
    print("CLI execution test passed.")
    
    # Load the generated synthetic samples
    gen_samples = pd.read_csv("/Users/noshinnawal/Downloads/dp-ctgans/dp_ctgans output.csv")
    
    # Check if the correct number of samples was generated
    assert len(gen_samples) == gen_size
    print("Synthetic sample size test passed.")

# Run the tests
test_dp_cgans()
test_cli()
