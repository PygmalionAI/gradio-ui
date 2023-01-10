def clear_stdout():
    '''
    Attempts to clear stdout, whether running in a notebook (IPython) or locally
    in a Unix envirnoment.
    '''
    try:
        from IPython.display import clear_output
        clear_output(wait=True)
    except ImportError:
        import os
        os.system("clear")
