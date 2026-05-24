

def extract_cifar10_resnet18_features(
    root="Real_Datasets/CIFAR10",
    cache_file="Real_Datasets/CIFAR10/cifar10_resnet18_features.npz",
    batch_size=256,
    num_workers=4,
    device=None,
):

    root = Path(root)
    cache_file = Path(cache_file)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    train_ds = CIFAR10(
        root=str(root),
        train=True,
        download=True,
        transform=transform,
    )

    test_ds = CIFAR10(
        root=str(root),
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    def _extract(loader):
        all_features = []

        with torch.inference_mode():
            for images, _ in tqdm(loader, desc="Extracting ResNet features"):
                images = images.to(device, non_blocking=True)
                feats = model(images)
                feats = feats.detach().cpu().numpy()
                all_features.append(feats)

        return np.vstack(all_features)

    features_train = _extract(train_loader)
    features_test = _extract(test_loader)

    labels_train = np.asarray(train_ds.targets, dtype=int)
    labels_test = np.asarray(test_ds.targets, dtype=int)

    np.savez_compressed(
        cache_file,
        features_train=features_train,
        labels_train=labels_train,
        features_test=features_test,
        labels_test=labels_test,
    )

    print(f"Saved features to: {cache_file}")
    print("features_train:", features_train.shape)
    print("features_test:", features_test.shape)

    return features_train, labels_train, features_test, labels_test



def load_cifar10_feature_cache(
    cache_file="Real_Datasets/CIFAR10/cifar10_resnet18_features.npz",
):
    data = np.load(cache_file)

    features_train = data["features_train"]
    labels_train = data["labels_train"]
    features_test = data["features_test"]
    labels_test = data["labels_test"]

    return features_train, labels_train, features_test, labels_test


def preprocess_features_for_occ(
    X_tr,
    X_cal,
    X_test,
    use_pca=True,
    pca_dim=50,
    random_state=None,
):

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_cal_s = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)

    if use_pca:
        pca_dim = min(pca_dim, X_tr_s.shape[1], X_tr_s.shape[0] - 1)

        pca = PCA(
            n_components=pca_dim,
            random_state=random_state,
        )

        X_tr_s = pca.fit_transform(X_tr_s)
        X_cal_s = pca.transform(X_cal_s)
        X_test_s = pca.transform(X_test_s)

    return X_tr_s, X_cal_s, X_test_s




CIFAR10_CLASS_NAMES = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def sample_stratified_indices(labels, classes, total_n, rng):

    labels = np.asarray(labels).reshape(-1)
    classes = list(classes)

    n_classes = len(classes)
    base = total_n // n_classes
    rem = total_n % n_classes

    selected = []

    for k, cls in enumerate(classes):
        idx = np.where(labels == cls)[0]
        idx = rng.permutation(idx)

        take = base + (1 if k < rem else 0)

        if take > len(idx):
            raise ValueError(
                f"Class {cls} only has {len(idx)} samples, "
                f"but requested {take}."
            )

        selected.append(idx[:take])

    selected = np.concatenate(selected)
    selected = rng.permutation(selected)

    return selected


def build_cifar_multiclass_ood_split(
    features_train,
    labels_train,
    features_test,
    labels_test,
    scenario,
    random_state=None,
):

    rng = np.random.default_rng(random_state)

    labels_train = np.asarray(labels_train).reshape(-1)
    labels_test = np.asarray(labels_test).reshape(-1)

    inlier_classes = scenario["inlier_classes"]
    outlier_classes = scenario["outlier_classes"]

    n_tr = int(scenario["n_tr"])
    n_cal = int(scenario["n_cal"])
    m = int(scenario["m"])
    pi_out = float(scenario["pi_out"])

    m_out = int(round(m * pi_out))
    m_null = m - m_out


    ref_idx = sample_stratified_indices(
        labels=labels_train,
        classes=inlier_classes,
        total_n=n_tr + n_cal,
        rng=rng,
    )

    tr_idx = ref_idx[:n_tr]
    cal_idx = ref_idx[n_tr:n_tr + n_cal]

    X_tr = features_train[tr_idx]
    X_cal = features_train[cal_idx]

    y_tr = labels_train[tr_idx]
    y_cal = labels_train[cal_idx]


    test_null_idx = sample_stratified_indices(
        labels=labels_test,
        classes=inlier_classes,
        total_n=m_null,
        rng=rng,
    )

    X_test_null = features_test[test_null_idx]
    y_test_null = labels_test[test_null_idx]


    test_out_idx = sample_stratified_indices(
        labels=labels_test,
        classes=outlier_classes,
        total_n=m_out,
        rng=rng,
    )

    X_test_out = features_test[test_out_idx]
    y_test_out = labels_test[test_out_idx]


    X_test = np.vstack([X_test_null, X_test_out])

    theta = np.concatenate([
        np.zeros(m_null, dtype=int),
        np.ones(m_out, dtype=int),
    ])

    y_test = np.concatenate([y_test_null, y_test_out])

    perm = rng.permutation(m)

    X_test = X_test[perm]
    theta = theta[perm]
    y_test = y_test[perm]

    meta = {
        "inlier_classes": inlier_classes,
        "outlier_classes": outlier_classes,
        "n_tr": n_tr,
        "n_cal": n_cal,
        "m": m,
        "m_null": m_null,
        "m_out": m_out,
        "pi_out": pi_out,
        "tr_idx": tr_idx,
        "cal_idx": cal_idx,
        "test_null_idx": test_null_idx,
        "test_out_idx": test_out_idx,
        "y_tr": y_tr,
        "y_cal": y_cal,
        "y_test": y_test,
    }

    return X_tr, X_cal, X_test, theta, meta

