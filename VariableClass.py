import inspect

class Variable:
    def __init__(self, name: str, function : callable):
        """
        Représente une variable dans la simulation, avec un nom et une fonction de calcul..

        param:
        -name : nom de la variable, par exemple "T" pour la température
        -function : fonction de calcul de la variable -> float/int, par exemple lambda x: 2*x + 1
        """
        self.name = name
        self.function = function

    def __str__(self):
        sig = inspect.signature(self.function)
        text = f"{self.name} = {self.function.__name__}({', '.join(list(sig.parameters.keys()))})\n"
        for param in sig.parameters.keys():
            text += f"  - {param} ∈ {sig.parameters[param].annotation}\n"
        return text

class Constant(Variable):
    def __init__(self, name: str, value: float | int):
        """
        Représente une constante dans la simulation, avec un nom et une valeur.

        param:
        -name : nom de la constante, par exemple "g" pour l'accélération gravitationnelle
        -value : valeur de la constante, par exemple 9.81
        """
        super().__init__(name, lambda: value)

    def __str__(self):
        return f"{self.name} = {self.function()}"

# Exemple d'utilisation
if __name__ == "__main__":
    def f(x: float, y: int) -> float:
        return x * y + 2

    var = Variable("ExampleVar", f)
    print(var)

    const = Constant("ExampleConst", 5.0)
    print(const)