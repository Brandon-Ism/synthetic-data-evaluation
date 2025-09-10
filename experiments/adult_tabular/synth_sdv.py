
import pandas as pd
from typing import Literal, Tuple
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

def train_and_sample_sdv(
    df_train: pd.DataFrame,
    n_samples: int,
    synth: Literal["ctgan", "tvae"] = "ctgan",
    epochs: int = 300,
    batch_size: int = 512,
    verbose: bool = False,
    enforce_rounding: bool = True,
    cuda: bool | None = None,
    pac: int = 10,   
) -> Tuple[pd.DataFrame, dict]:

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)
    try:
        metadata.set_table_name("adult")
    except Exception:
        pass

    common_args = dict(
        metadata=metadata,
        epochs=epochs,
        enforce_rounding=enforce_rounding,
        verbose=verbose,
        batch_size=batch_size,
    )
    if synth == "ctgan":
        model = CTGANSynthesizer(**common_args, pac=pac)  
    elif synth == "tvae":
        model = TVAESynthesizer(**common_args)
    else:
        raise ValueError(f"Unknown synthesizer: {synth}")

    model.fit(df_train)
    sampled = model.sample(num_rows=n_samples)

    info = {
        "library": "sdv",
        "synth": synth.upper(),
        "epochs": epochs,
        "batch_size": batch_size,
        "enforce_rounding": enforce_rounding,
        "pac": pac,
    }
    return sampled, info
