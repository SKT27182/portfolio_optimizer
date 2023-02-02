class Style:
    WHITE = "\033[37m"
    HEADER = '\033[95m'
    BLUE = "\033[34m"
    OKBLUE = '\033[94m'
    LIGHT_BLUE = '\033[94m'
    GREEN = "\033[32m"
    LIGHT_GREEN = '\033[92m'
    OKGREEN = '\033[92m'
    BLACK = "\033[30m"
    RED = "\033[31m"
    LIGHT_RED = '\033[91m'
    YELLOW = '\033[33m'
    LIGHT_YELLOW = '\033[93m'
    CYAN = '\033[36m'
    LIGHT_CYAN = '\033[96m'
    MAGENTA = '\033[35m'
    LIGHT_MAGENTA = '\033[95m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    

    

class cprint(Style):

    """
    Available options are:

    - WHITE 
    - HEADER
    - BLUE 
    - OKBLUE
    - LIGHT_BLUE 
    - GREEN 
    - LIGHT_GREEN
    - OKGREEN 
    - BLACK 
    - RED 
    - LIGHT_RED 
    - YELLOW
    - LIGHT_YELLO 
    - CYAN 
    - LIGHT_CYAN 
    - MAGENTA 
    - LIGHT_MAGEN
    - WARNING 
    - FAIL 
    - ENDC 
    - BOLD 
    - UNDERLINE 

    """


    @staticmethod
    def print(text, color, *args, **kwargs):
        color = color.upper()
        print(eval(f"Style.{color}") + str(text) + Style.ENDC, *args, **kwargs)
