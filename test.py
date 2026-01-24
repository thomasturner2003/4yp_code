def fair_value(combined, value_1, value_2):
    return combined*(value_1/(value_2+value_1)), combined*(value_2/(value_2+value_1))

print(fair_value(2.5, 1.3, 0.8))