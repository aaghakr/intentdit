# Experiment Names Consistency Check

## ✅ All 24 Experiments - Consistent Naming Convention

### **Naming Pattern:**
`{dataset}_{encoder}_{spatial_guidance}[_text]`

Where:
- `dataset`: `pku` or `cgl`
- `encoder`: `vit` or `swin`
- `spatial_guidance`: `saliency`, `intent`, or `both`
- `text`: `_text` (only when text control is enabled)

### **Complete List of 24 Experiments:**

#### **PKU Dataset (12 experiments):**

**ViT Encoder (6 experiments):**
1. `pku_vit_saliency` - PKU dataset, ViT encoder, saliency only
2. `pku_vit_intent` - PKU dataset, ViT encoder, intent only
3. `pku_vit_both` - PKU dataset, ViT encoder, saliency + intent
4. `pku_vit_saliency_text` - PKU dataset, ViT encoder, saliency + text
5. `pku_vit_intent_text` - PKU dataset, ViT encoder, intent + text
6. `pku_vit_both_text` - PKU dataset, ViT encoder, saliency + intent + text

**Swin Encoder (6 experiments):**
7. `pku_swin_saliency` - PKU dataset, Swin encoder, saliency only
8. `pku_swin_intent` - PKU dataset, Swin encoder, intent only
9. `pku_swin_both` - PKU dataset, Swin encoder, saliency + intent
10. `pku_swin_saliency_text` - PKU dataset, Swin encoder, saliency + text
11. `pku_swin_intent_text` - PKU dataset, Swin encoder, intent + text
12. `pku_swin_both_text` - PKU dataset, Swin encoder, saliency + intent + text

#### **CGL Dataset (12 experiments):**

**ViT Encoder (6 experiments):**
13. `cgl_vit_saliency` - CGL dataset, ViT encoder, saliency only
14. `cgl_vit_intent` - CGL dataset, ViT encoder, intent only
15. `cgl_vit_both` - CGL dataset, ViT encoder, saliency + intent
16. `cgl_vit_saliency_text` - CGL dataset, ViT encoder, saliency + text
17. `cgl_vit_intent_text` - CGL dataset, ViT encoder, intent + text
18. `cgl_vit_both_text` - CGL dataset, ViT encoder, saliency + intent + text

**Swin Encoder (6 experiments):**
19. `cgl_swin_saliency` - CGL dataset, Swin encoder, saliency only
20. `cgl_swin_intent` - CGL dataset, Swin encoder, intent only
21. `cgl_swin_both` - CGL dataset, Swin encoder, saliency + intent
22. `cgl_swin_saliency_text` - CGL dataset, Swin encoder, saliency + text
23. `cgl_swin_intent_text` - CGL dataset, Swin encoder, intent + text
24. `cgl_swin_both_text` - CGL dataset, Swin encoder, saliency + intent + text

## ✅ Files Updated for Consistency:

### **1. Demo Configuration (`demo/demo_config.yaml`)**
- ✅ All 24 experiments defined with consistent naming
- ✅ Checkpoint paths follow the naming convention
- ✅ Model weights properly configured

### **2. Demo App (`demo/app.py`)**
- ✅ Experiment status display uses consistent names
- ✅ Configuration loader uses consistent names
- ✅ All 24 experiments listed in interface

### **3. README (`code/README.md`)**
- ✅ Training commands use consistent experiment names
- ✅ Testing commands use consistent experiment names
- ✅ Experimental design summary uses consistent names

### **4. All Experiments Script (`code/all_experiments.sh`)**
- ✅ All 24 training commands use consistent names
- ✅ GPU allocation follows the naming convention
- ✅ Error handling uses consistent names

### **5. Demo App Interface**
- ✅ Experiment status display shows consistent names
- ✅ Configuration validation uses consistent names
- ✅ All comparison types use consistent names

## ✅ Verification Checklist:

- [x] **Demo Config**: All 24 experiments defined
- [x] **Demo App**: All 24 experiments displayed
- [x] **README**: All 24 experiments documented
- [x] **All Experiments Script**: All 24 experiments automated
- [x] **Naming Convention**: Consistent across all files
- [x] **Spatial Guidance**: `saliency`, `intent`, `both` (not `sal`, `sal_intent`)
- [x] **Text Control**: `_text` suffix (not `_text` in middle)
- [x] **Dataset Names**: `pku`, `cgl` (consistent)
- [x] **Encoder Names**: `vit`, `swin` (consistent)

## 🎯 Key Changes Made:

### **Before (Inconsistent):**
- `pku_vit_sal` → `pku_vit_saliency`
- `pku_vit_sal_intent` → `pku_vit_both`
- `pku_vit_sal_intent_text` → `pku_vit_both_text`
- `cgl_swin_sal_intent_text` → `cgl_swin_both_text`

### **After (Consistent):**
- All experiments follow: `{dataset}_{encoder}_{spatial_guidance}[_text]`
- Spatial guidance: `saliency`, `intent`, `both`
- Text control: `_text` suffix only
- All 24 experiments consistent across all files

## ✅ Final Status:
**ALL 24 EXPERIMENT NAMES ARE NOW CONSISTENT ACROSS ALL FILES!**

The naming convention is now standardized and all files (demo config, README, scripts, and app) use the same experiment names.
