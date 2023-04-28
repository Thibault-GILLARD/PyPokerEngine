def encode_card(card_str):
    rank_str = '23456789TJQK1'
    suit_str = 'cdhs'

    # Extract rank and suit from the card string
    suit_char, rank_char = card_str[0], card_str[1]

    # Find the rank and suit representation
    rank_representation = rank_str.index(rank_char)
    suit_representation = suit_str.index(suit_char)

    # Combine the rank and suit representations
    encoded_card = rank_representation * 4 + suit_representation

    return encoded_card

# Example usage:

encoded_card = encode_card("sj")
print(encoded_card)  # Output: 18