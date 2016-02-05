class initialisationError(Exception):
    def __init__(self, value):
         self.value = value
         #self.msg =msg
    def __str__(self):
        return repr(self.value)
