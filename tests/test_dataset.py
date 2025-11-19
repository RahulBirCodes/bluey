from scripts.dataset import get_batch

def test_get_batch():
    B = 2
    T = 1
    D = 2
    tokens, X, Y, W, x_pos = get_batch(
        batch_size=B,
        num_pairs=T,
        xy_size=D,
        device="cpu"
    )
    print("\n=== X ===")
    print(X)
    print("\n=== Y ===")
    print(Y)
    print("\n=== x_pos (positions of X tokens) ===")
    print(x_pos)
    print("\n=== tokens ===")
    for b in range(B):
        print(f"\nBatch element {b}:")
        print(tokens[b])

if __name__ == "__main__":
    test_get_batch()