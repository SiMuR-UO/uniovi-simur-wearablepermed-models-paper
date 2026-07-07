import sys
import argparse
import logging
import subprocess
import pandas as pd
from enum import Enum
from datetime import datetime

__author__ = "Miguel Angel Salinas Gancedo<uo34525@uniovi.es>, Alejandro Castellanos Alonso<uo265351@uniovi.es>, Antonio Miguel López Rodriguez<amlopez@uniovi.es>"
__copyright__ = "Uniovi"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

class Model_Type(Enum):
    INDIVIDUAL = 'Individual'
    FUSION_CONCATENATED = 'Fusion Concatenated'
    FUSION_STACK_RF = 'Fusion Stack RF'
    FUSION_STACK_AE = 'Fusion Stack AE'
    FUSION_MOE_RF = 'Fusion MoE RF'
    FUSION_MOE_AE = 'Fusion MoE AE'

class Superclasses(Enum):
    CPA_METS = 'CPA-METS'
    CAPTURED_24 = 'Captured24'
    WEARABLE_PER_MED = 'WearablePerMed'

class Segment_Bodies_Combinations(Enum):
    PI = 'Thigh'
    M = 'Wrist'    
    C = 'Hip'
    PI_M = 'Thigh + Wrist'
    PI_C = 'Thigh + Hip'
    M_C = 'Wrist + Hip'
    PI_M_C = 'Thigh + Wrist + Hip'

def parse_args(args):
    """Parse command line parameters

    Args:hip
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="WareablePerMed Pipeline")
    parser.add_argument(
        "-stack-folder",
        "--stack-folder",
        dest="stack_folder",       
        required=True,
        help=f"Stack folder with pretrain datasets to be trained."
    )    
    parser.add_argument(
        "-model-types",
        "--model-types",
        dest="model_types", 
        nargs='+',
        choices=[c.value for c in Model_Type],        
        help=f"Model Type Strategies to be trained: {[c.value for c in Model_Type]}."
    )    
    parser.add_argument(
        "-superclasses",
        "--superclasses",
        dest="superclasses",
        nargs='+',
        help=f"Superclasses to be train: {[c.value for c in Superclasses]}."
    )
    parser.add_argument(
        "-segment-bodies-combinations",
        "--segment-bodies-combinations",
        dest="segment_bodies_combinations",
        nargs='+',                    
        help=f"Segment bodies combination to be trained: {[c.value for c in Segment_Bodies_Combinations]}."
    )           
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO.",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG.",
        action="store_const",
        const=logging.DEBUG,
    )  

    return parser.parse_args(args)

def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"

    logging.basicConfig(
        level=loglevel,
        stream=sys.stdout,
        format=logformat,
        datefmt="%Y-%m-%d %H:%M:%S"
    )    

def execute_command(command):
    # Redirect stderr to stdout so they share a single, safe stream
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True
    )

    all_lines = []
    
    # This reads everything (prints and errors) in real-time as it happens
    for line in process.stdout:
        line = line.strip()
        all_lines.append(line)
        _logger.info("Processing: " + line)

    # Wait for the process to actually exit
    process.wait()

    full_output_text = "\n".join(all_lines)

    # Check exit code
    if process.returncode != 0:
        error_msg = (
            f"Command failed with exit code {process.returncode}\n"
            f"Command: {' '.join(command)}\n"
            f"OUTPUT:\n{full_output_text}\n"
        )

        _logger.error(error_msg)

        raise subprocess.CalledProcessError(
            returncode=process.returncode,
            cmd=command,
            output=error_msg
        )
        
    return full_output_text
    
def train_individual_strategy(stack_all, superclases, segment_body):
    script = "1_individual_rf_loocv.py"

    cmd = [
        sys.executable,
        "src/" + script,
        "--stack-all", stack_all,
        "--segment-body", segment_body,
        "--superclases", superclases
    ]

    _logger.info(f"Command {cmd}")

    return execute_command(cmd) 

def train_fusion_concatenated_strategy(stack_all, superclases):
    script = "2_concatenate_rf_PI_M.py"

    cmd = [
        sys.executable,
        "src/" + script,
        "--stack-all", stack_all,
        "--superclases", superclases
    ]

    _logger.info(f"Command {cmd}")

    return execute_command(cmd) 

def train_fusion_stack_rf_2_segment_bodies_strategy(stack_all, superclases):
    script = "3_stack_rf_PI_M_loocv.py"

    cmd = [
        sys.executable,
        "src/" + script,
        "--stack-all", stack_all,
        "--superclases", superclases
    ]

    _logger.info(f"Command {cmd}")

    return execute_command(cmd) 

def train_fusion_stack_ae_2_segment_bodies_strategy(stack_all, superclases):
    script = "3_stack_ae_PI_M_loocv.py"

    cmd = [
        sys.executable,
        "src/" + script,
        "--stack-all", stack_all,
        "--superclases", superclases
    ]

    _logger.info(f"Command {cmd}")

    return execute_command(cmd) 

def train_fusion_stack_rf_3_segment_bodies_strategy(stack_all, superclases):
    script = "3_stack_rf_PI_M_C_loocv.py"

    cmd = [
        sys.executable,
        "src/" + script,
        "--stack-all", stack_all,
        "--superclases", superclases
    ]

    _logger.info(f"Command {cmd}")

    return execute_command(cmd) 

def train_fusion_stack_ae_3_segment_bodies_strategy(stack_all, superclases):
    script = "3_stack_ae_PI_M_C_loocv.py"

    cmd = [
        sys.executable,
        "src/" + script,
        "--stack-all", stack_all,
        "--superclases", superclases
    ]

    _logger.info(f"Command {cmd}")

    return execute_command(cmd) 

def train_fusion_moe_rf_2_segment_bodies_strategy(stack_all, superclases):
    script = "4_mixture_of_experts_rf_PI_M_loocv.py"

    cmd = [
        sys.executable,
        "src/" + script,
        "--stack-all", stack_all,
        "--superclases", superclases
    ]

    _logger.info(f"Command {cmd}")

    return execute_command(cmd) 

def train_fusion_moe_ae_2_segment_bodies_strategy(stack_all, superclases):
    script = "4_mixture_of_experts_ae_PI_M_loocv.py"

    cmd = [
        sys.executable,
        "src/" + script,
        "--stack-all", stack_all,
        "--superclases", superclases
    ]

    _logger.info(f"Command {cmd}")

    return execute_command(cmd) 

def train_fusion_moe_rf_3_segment_bodies_strategy(stack_all, superclases):
    script = "4_mixture_of_experts_rf_PI_M_C_loocv.py"

    cmd = [
        sys.executable,
        "src/" + script,
        "--stack-all", stack_all,
        "--superclases", superclases
    ]

    _logger.info(f"Command {cmd}")

    return execute_command(cmd) 

def train_fusion_moe_ae_3_segment_bodies_strategy(stack_all, superclases):
    script = "4_mixture_of_experts_ae_PI_M_C_loocv.py"

    cmd = [
        sys.executable,
        "src/" + script,
        "--stack-all", stack_all,
        "--superclases", superclases
    ]

    _logger.info(f"Command {cmd}")

    return execute_command(cmd) 

def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)

    _logger.info("Pipeline starts here at " + str(datetime.now()))

    if (args.segment_bodies_combinations is None):
        segment_bodies_combinations = list(Segment_Bodies_Combinations)
    else:
        segment_bodies_combinations = [Segment_Bodies_Combinations(v) for v in args.segment_bodies_combinations]

    if (args.superclasses is None):
        superclasses = list(Superclasses)
    else:
        superclasses = [Superclasses(v) for v in args.superclasses]

    if (args.model_types is None):
        model_types = list(Model_Type)
    else:
        model_types = [Model_Type(v) for v in args.model_types]

    for segment_body in segment_bodies_combinations:
        for superclass in superclasses:
            # construct the stack input folder from segment body selected
            stack_all = args.stack_folder + "/case_" + segment_body.name + "_BRF_acc_gyr_15_classes/data_feature_all.npz"

            for model_type in model_types:
                _logger.info(f"Execute pipeline for Model Type: {model_type.value}; Superclass: {superclass.value}; Segment Body : {segment_body.value}")

                if (model_type.value in (Model_Type.INDIVIDUAL.value)):
                    train_individual_strategy(stack_all, superclass.value, segment_body.name)
                elif (model_type.value in (Model_Type.FUSION_CONCATENATED.value)):
                    train_fusion_concatenated_strategy(stack_all, superclass.value)
                elif (model_type.value in (Model_Type.FUSION_STACK_RF.value)):  
                    if (segment_body.name in (Segment_Bodies_Combinations.PI_M, Segment_Bodies_Combinations.PI_C, Segment_Bodies_Combinations.M_C)):
                        train_fusion_stack_rf_2_segment_bodies_strategy(stack_all, superclass.value)
                    else:
                        train_fusion_stack_rf_3_segment_bodies_strategy(stack_all, superclass.value)
                elif (model_type.value in (Model_Type.FUSION_STACK_AE.value)):
                    if (segment_body.name in (Segment_Bodies_Combinations.PI_M, Segment_Bodies_Combinations.PI_C, Segment_Bodies_Combinations.M_C)):
                        train_fusion_stack_ae_2_segment_bodies_strategy(stack_all, superclass.value)
                    else:
                        train_fusion_stack_ae_3_segment_bodies_strategy(stack_all, superclass.value)
                elif (model_type.value in (Model_Type.FUSION_MOE_RF.value)):
                    if (segment_body.name in (Segment_Bodies_Combinations.PI_M, Segment_Bodies_Combinations.PI_C, Segment_Bodies_Combinations.M_C)):
                        train_fusion_moe_rf_2_segment_bodies_strategy(stack_all, superclass.value)
                    else:
                        train_fusion_moe_rf_3_segment_bodies_strategy(stack_all, superclass.value)
                elif (model_type.value in (Model_Type.FUSION_MOE_AE.value)):
                    if (segment_body.name in (Segment_Bodies_Combinations.PI_M, Segment_Bodies_Combinations.PI_C, Segment_Bodies_Combinations.M_C)):
                        train_fusion_moe_ae_2_segment_bodies_strategy(stack_all, superclass.value)
                    else:
                        train_fusion_moe_ae_3_segment_bodies_strategy(stack_all, superclass.value)                    

    df_error = pd.DataFrame(columns=["model_type", "command"])

    # Start pre training pipelines for all participants 
    _logger.info("Start pre training pipelines for all participants")

    # Open the file for writing
    df_error.to_csv("error_log.csv", index=False)

    _logger.info("Pipeline ends here at " + str(datetime.now()))

def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])

if __name__ == "__main__":
        # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m wearablepermed_utils.skeleton 42
    #
    run()