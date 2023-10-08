import math
class Fraction:
    def __init__(self,numerator,denominator):
        self.numerator = numerator
        self.denominator = denominator
        # local
        self._reduce()

    def _reduce(self):
        gcd = math.gcd(self.numerator, self.denominator)
        self.numerator = self.numerator // gcd 
        self.denominator = self.denominator // gcd
        # if < -1

    def __repr__(self):
        string = str(self.numerator) + "/" + str(self.denominator)
        return string
    
    def __eq__(self,other):
        if self.numerator == other.numerator:
            if self.denominator == other.denominator:
                return True
        return False
    
    def __add__(self,other):
        denomintor = self.denominator*other.denominator
        numerator = self.numerator*other.denominator + self.denominator*other.numerator
        return Fraction(numerator, denomintor)
    # def __sub__(self, other)
    # def __mul__(self, other)
    # def __neg__(self, other)
    # def _
    @staticmethod
    def from_string(string):
        n,d = string.split("/")
        n = int(n)
        d = int(d)
        return Fraction(n,d)

a = Fraction(1,2)
b = Fraction(2,4)
c = Fraction.from_string("1/2")
print(a==b)
print(c)
print(a)
print(a+b)