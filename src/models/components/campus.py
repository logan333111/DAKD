import os
import hashlib
from urllib.parse import urlparse

import torch
from torch import nn
import torchvision.models as models

import open_clip

feature_norm = lambda x: x / (x.norm(dim=-1, keepdim=True) + 1e-10)

_STUDENT_ARCH_ALIASES = {
    "eff-b0": "efficientnet_b0",
    "shuf-v2": "shufflenet_v2_x1_0",
    "swin-t": "swin_t",
    "swin-tiny": "swin_t",
    "swin_tiny": "swin_t",
}

_STUDENT_WEIGHT_ENUMS = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "mobilenet_v2": "MobileNet_V2_Weights",
    "efficientnet_b0": "EfficientNet_B0_Weights",
    "shufflenet_v2_x1_0": "ShuffleNet_V2_X1_0_Weights",
    "swin_t": "Swin_T_Weights",
}

_KNOWN_WEIGHT_HASH_FALLBACKS = {
    # Official PyTorch Vision has a long-standing mismatch for EfficientNet-B0:
    # the URL name encodes 3dd342df while the served file hashes to 7f5810bc...
    "efficientnet_b0": {
        "7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934",
    }
}


def _make_model_without_weights(model_fn):
    try:
        return model_fn(weights=None)
    except TypeError:
        return model_fn(pretrained=False)


def _sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_with_known_hash_fallback(arch, model_fn, weights_enum):
    url = weights_enum.DEFAULT.url
    state_dict = torch.hub.load_state_dict_from_url(url, check_hash=False, progress=True)
    cached_file = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(urlparse(url).path))
    digest = _sha256_file(cached_file)
    allowed_digests = _KNOWN_WEIGHT_HASH_FALLBACKS.get(arch, set())
    if digest not in allowed_digests:
        raise RuntimeError(
            f"Downloaded pretrained weights for '{arch}' but the file hash '{digest}' is not in the "
            f"trusted fallback list. Expected one of: {sorted(allowed_digests)}"
        )

    model = _make_model_without_weights(model_fn)
    model.load_state_dict(state_dict)
    return model


def build_student_backbone(arch, pretrained=True):
    requested_arch = arch
    arch = _STUDENT_ARCH_ALIASES.get(arch, arch)
    if arch not in _STUDENT_WEIGHT_ENUMS:
        supported = ", ".join(sorted(_STUDENT_WEIGHT_ENUMS))
        aliases = ", ".join(f"{k}->{v}" for k, v in sorted(_STUDENT_ARCH_ALIASES.items()))
        raise ValueError(
            f"Unsupported student arch '{requested_arch}'. Supported student backbones: {supported}. Aliases: {aliases}"
        )

    model_fn = getattr(models, arch, None)
    if model_fn is None:
        raise ValueError(
            f"Student arch '{requested_arch}' resolves to '{arch}', but this torchvision version "
            f"does not provide torchvision.models.{arch}. Upgrade torchvision or choose another backbone."
        )

    if not pretrained:
        weights_enum_name = _STUDENT_WEIGHT_ENUMS[arch]
        if hasattr(models, weights_enum_name):
            return model_fn(weights=None)
        return model_fn(pretrained=False)

    weights_enum_name = _STUDENT_WEIGHT_ENUMS[arch]
    weights_enum = getattr(models, weights_enum_name, None)
    if weights_enum is not None:
        try:
            return model_fn(weights=weights_enum.DEFAULT)
        except RuntimeError as e:
            checkpoints_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
            msg = str(e)
            if "invalid hash value" in msg:
                if arch in _KNOWN_WEIGHT_HASH_FALLBACKS:
                    return _load_with_known_hash_fallback(arch, model_fn, weights_enum)
                raise RuntimeError(
                    f"Failed to load pretrained weights for student arch '{requested_arch}' "
                    f"(resolved to '{arch}'). Torchvision reported a hash mismatch, which usually "
                    f"means the downloaded cache file is corrupted or the download returned the wrong content. "
                    f"Clear the cached student weights under '{checkpoints_dir}' and retry, "
                    f"or run with 'model.net.student.pretrained=false' to start from random initialization."
                ) from e
            raise
    return model_fn(pretrained=True)


def get_classifier_in_features(model):
    if hasattr(model, "fc") and hasattr(model.fc, "in_features"):
        return model.fc.in_features
    if hasattr(model, "head") and hasattr(model.head, "in_features"):
        return model.head.in_features
    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential):
            for module in reversed(classifier):
                if hasattr(module, "in_features"):
                    return module.in_features
        if hasattr(classifier, "in_features"):
            return classifier.in_features
    raise ValueError(f"Unsupported student head type for '{model.__class__.__name__}'")


def replace_classifier(model, new_head):
    if hasattr(model, "fc") and hasattr(model.fc, "in_features"):
        model.fc = new_head
        return
    if hasattr(model, "head") and hasattr(model.head, "in_features"):
        model.head = new_head
        return
    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential):
            modules = list(classifier)
            for idx in range(len(modules) - 1, -1, -1):
                if hasattr(modules[idx], "in_features"):
                    modules[idx] = new_head
                    model.classifier = nn.Sequential(*modules)
                    return
        if hasattr(classifier, "in_features"):
            model.classifier = new_head
            return
    raise ValueError(f"Unsupported student head type for '{model.__class__.__name__}'")

class TeacherStudent(nn.Module):
    def __init__(self, teacher, student, data_attributes, use_teacher=True):
        super(TeacherStudent, self).__init__()
        self.teacher, self.align, self.frozen_nlp_features = None, None, None
        self.data_attributes = data_attributes
        self.student = StudentNet(student, data_attributes.class_num, use_teacher)
        if use_teacher:
            device = next(self.student.model.resnet.parameters()).device
            self.teacher = TeacherNet(teacher)
            self.align = AlignNet(self.teacher.last_features_dim, self.student.num_features)
            self.frozen_nlp_features = self.get_frozen_nlp_features(data_attributes)
    
    def get_frozen_nlp_features(self, attributes):
        prompt_tmpl = attributes.prompt_tmpl
        classes_list = list(attributes.classes.values())
        text_tokens = self.teacher.tokenizer([prompt_tmpl.format(word) for word in classes_list])
        nlp_features = self.teacher.encode_text(text_tokens).detach()
        return feature_norm(nlp_features)
    
    def forward(self, x):
        if self.teacher:
            clip_img_features = self.teacher(x)
            frozen_nlp_features = self.frozen_nlp_features.to(clip_img_features.device)
            aligned_img, aligned_nlp = self.align(clip_img_features, frozen_nlp_features)
            hidden_features, out = self.student(x)
            return hidden_features, out, clip_img_features, frozen_nlp_features, aligned_img, aligned_nlp
        return self.student(x)



class TeacherNet(nn.Module):
    def __init__(self, teacher):
        super(TeacherNet, self).__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(teacher.arch, pretrained=teacher.pretrained)
        self.model.requires_grad_(False)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(teacher.arch)
        self.last_features_dim = self.model.transformer.resblocks[-1].mlp.c_proj.out_features

    def encode_image(self, x):
        return self.model.encode_image(x)

    def encode_text(self, x):
        return self.model.encode_text(x)

    def forward(self, x):
        with torch.no_grad():
            clip_img_features = self.encode_image(x).detach()
        clip_img_features = feature_norm(clip_img_features)
        return clip_img_features
    

class AlignNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(AlignNet, self).__init__()

        self.align_img_layer = nn.Sequential(
            nn.Linear(in_features, out_features), 
            nn.ReLU(), 
            nn.Linear(out_features, out_features)
            )
        self.align_nlp_layer = nn.Sequential(
            nn.Linear(in_features, out_features), 
            nn.ReLU(), 
            nn.Linear(out_features, out_features)
        )
    
    def forward(self, x, clip_nlp_features):
        align_img = self.align_img_layer(x)
        align_nlp = self.align_nlp_layer(clip_nlp_features)
        return feature_norm(align_img), feature_norm(align_nlp)



class StudentNet(nn.Module):
    def __init__(self, student, class_num, use_teacher=True):
        super(StudentNet, self).__init__()
        self.use_teacher = use_teacher
        self.num_features = None
        pretrained = getattr(student, "pretrained", True)
        if self.use_teacher:
            origin_model = build_student_backbone(student.arch, pretrained=pretrained)
            self.model  = ModifiedResNet(origin_model, class_num)
            self.num_features = self.model.num_features
        else:
            self.model = build_student_backbone(student.arch, pretrained=pretrained)
            num_features = get_classifier_in_features(self.model)
            replace_classifier(self.model, nn.Linear(num_features, class_num))

    def forward(self, x):
        if self.use_teacher:
            hidden_features, out = self.model(x)
            return feature_norm(hidden_features), out
        out = self.model(x)
        return out


class ModifiedResNet(torch.nn.Module):
    def __init__(self, origin_model, classnum):
        super(ModifiedResNet, self).__init__()
        # Keep `resnet` for checkpoint compatibility with existing student keys.
        self.resnet = origin_model
        num_features = get_classifier_in_features(origin_model)
        replace_classifier(self.resnet, nn.Identity())
        self.linear_cls = nn.Linear(num_features, classnum)
        self.num_features = num_features

    def forward(self, x):
        hidden_features = self.resnet(x)
        out = self.linear_cls(hidden_features)

        return hidden_features, out





if __name__ == "__main__":
    _ = TeacherStudent()
