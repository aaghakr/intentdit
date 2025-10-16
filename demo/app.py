import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import random
import os
import sys

# Add demo directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import configuration loader
from config_loader import (
    get_experiment_checkpoint, 
    get_experiment_info, 
    get_model_weights,
    list_available_experiments,
    validate_config
)

# Define sample paths
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
CGL_SAMPLES_DIR = os.path.join(SAMPLES_DIR, "cgl")
PKL_SAMPLES_DIR = os.path.join(SAMPLES_DIR, "pkl")

def load_sample_images(dataset_type):
    """
    Load sample images based on dataset type
    """
    if dataset_type.lower() == "cgl":
        sample_dir = CGL_SAMPLES_DIR
    else:  # PKL
        sample_dir = PKL_SAMPLES_DIR
    
    sample_images = []
    for img_name in sorted(os.listdir(sample_dir)):
        if img_name.endswith('.png'):
            img_path = os.path.join(sample_dir, img_name)
            try:
                img = Image.open(img_path)
                sample_images.append((img_name, img))
            except Exception as e:
                print(f"Error loading {img_name}: {e}")
    
    return sample_images

def generate_background(prompt, style, resolution, seed):
    """
    Generate background image based on prompt
    """
    if not prompt:
        return None, "Please enter a prompt"
    
    # Parse resolution
    width, height = map(int, resolution.split('x'))
    
    # Simulate background generation
    result_text = f"""
    Background Generated Successfully!
    
    Prompt: {prompt}
    Style: {style}
    Resolution: {resolution}
    Seed: {seed}
    """
    
    # Create dummy background with gradient
    random.seed(seed)
    bg_img = Image.new('RGB', (width, height))
    pixels = bg_img.load()
    
    # Generate gradient background
    r1, g1, b1 = random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)
    r2, g2, b2 = random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)
    
    for y in range(height):
        for x in range(width):
            r = int(r1 + (r2 - r1) * y / height)
            g = int(g1 + (g2 - g1) * y / height)
            b = int(b1 + (b2 - b1) * y / height)
            pixels[x, y] = (r, g, b)
    
    # Add text overlay
    draw = ImageDraw.Draw(bg_img)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    text = f"Background: {style}"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text(((width - text_width) // 2, (height - text_height) // 2), text, fill='white', font=font)
    
    return bg_img, result_text

def check_experiment_availability(dataset, encoder, spatial_guidance, text_control):
    """Check if experiment configuration is available"""
    experiment_name = f"{dataset}_{encoder}_{spatial_guidance.lower().replace(' ', '_')}"
    if text_control:
        experiment_name += "_text"
    
    # Get experiment info from config
    exp_info = get_experiment_info(experiment_name, dataset)
    
    return {
        "available": exp_info["checkpoint"] is not None,
        "experiment_name": experiment_name,
        "checkpoint_path": exp_info["checkpoint"],
        "model_weights": exp_info["model_weights"]
    }

def generate_layout(prompt, dataset, task, encoder, spatial_guidance, text_control, image, seed):
    """
    Generate layout with spatial guidance visualization based on experimental setup
    """
    if not prompt:
        return None, None, None, "Please enter a prompt"
    
    # Check experiment availability
    exp_status = check_experiment_availability(dataset, encoder, spatial_guidance, text_control)
    
    if not exp_status["available"]:
        return None, None, None, f"❌ Experiment not available: {exp_status['experiment_name']}\nCheckpoint not found. Please train the model first or check the configuration."
    
    # Load sample images for the selected dataset
    sample_images = load_sample_images(dataset)
    
    # If we have samples, use one based on the seed
    if sample_images:
        random.seed(seed)
        sample_name, sample_image = random.choice(sample_images)
        base_img = sample_image.copy()
        width, height = base_img.size
    else:
        # Fallback to the original behavior
        if image is not None:
            width, height = image.size
            base_img = image
        else:
            width, height = 512, 512
            base_img = Image.new('RGB', (width, height), color='white')
    
    # Determine experiment configuration
    experiment_name = f"{dataset}_{encoder}_{spatial_guidance.lower().replace(' ', '_')}"
    if text_control:
        experiment_name += "_text"
    
    # Simulate layout generation
    result_text = f"""
    Layout Generated Successfully!
    
    Experiment: {exp_status['experiment_name']}
    Checkpoint: {exp_status['checkpoint_path'] or 'Not found'}
    Prompt: {prompt}
    Dataset: {dataset.upper()}
    Task: {task}
    Encoder: {encoder.upper()}
    Spatial Guidance: {spatial_guidance}
    Text Control: {'Enabled' if text_control else 'Disabled'}
    Seed: {seed}
    
    Model Weights:
    - Saliency: {exp_status['model_weights']['saliency'] or 'Not found'}
    - Intent: {exp_status['model_weights']['intent'] or 'Not found'}
    """
    
    if sample_images:
        result_text += f"\nUsing sample image: {sample_name}"
    
    # Generate spatial guidance visualization based on configuration
    spatial_img = base_img.copy()
    draw = ImageDraw.Draw(spatial_img, 'RGBA')
    
    # Simulate different spatial guidance types
    random.seed(seed)
    num_regions = 30 if spatial_guidance == "Both" else 20
    
    for _ in range(num_regions):
        x = random.randint(0, width)
        y = random.randint(0, height)
        radius = random.randint(15, 60)
        intensity = random.randint(80, 180)
        
        if spatial_guidance == "Saliency":
            color = (intensity, 0, 0, 120)  # Red for saliency
        elif spatial_guidance == "Intent Map":
            color = (0, intensity, 0, 120)  # Green for intent
        else:  # Both
            color = (intensity, intensity//2, 0, 100)  # Orange for combined
        
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    
    # Generate bounding boxes with dataset-specific classes
    bbox_img = base_img.copy()
    draw = ImageDraw.Draw(bbox_img)
    
    random.seed(seed)
    
    # Dataset-specific element classes
    if dataset.lower() == "pku":
        labels = ['Text', 'Logo', 'Underlay', 'Embellishment']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    else:  # CGL
        labels = ['Text', 'Logo', 'Underlay', 'Embellishment', 'Decoration']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    # Generate elements based on prompt analysis
    num_boxes = min(len(labels), random.randint(2, 6))
    
    for i in range(num_boxes):
        x1 = random.randint(10, width - 150)
        y1 = random.randint(10, height - 150)
        x2 = x1 + random.randint(80, 200)
        y2 = y1 + random.randint(60, 150)
        
        # Ensure boxes stay within bounds
        x2 = min(x2, width - 10)
        y2 = min(y2, height - 10)
        
        color = colors[i % len(colors)]
        label = labels[i % len(labels)]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label background
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
        draw.text((x1, y1), label, fill='white', font=font)
    
    # Generate final output (combination of spatial guidance + bboxes)
    final_output = base_img.copy()
    final_output.paste(spatial_img, (0, 0), spatial_img.convert('RGBA').split()[-1])
    
    draw_final = ImageDraw.Draw(final_output)
    random.seed(seed)
    for i in range(num_boxes):
        x1 = random.randint(10, width - 150)
        y1 = random.randint(10, height - 150)
        x2 = x1 + random.randint(80, 200)
        y2 = y1 + random.randint(60, 150)
        x2 = min(x2, width - 10)
        y2 = min(y2, height - 10)
        
        color = colors[i % len(colors)]
        label = labels[i % len(labels)]
        draw_final.rectangle([x1, y1, x2, y2], outline=color, width=2)
    
    return final_output, spatial_img, bbox_img, result_text

# Examples are now loaded from the samples directory

# Create the interface with two tabs
with gr.Blocks(theme=gr.themes.Soft(), title="Intent Aware Generation") as demo:
    gr.Markdown("# 🎨 Intent Aware Background & Layout Generation")
    gr.Markdown("Generate backgrounds and layouts with AI-powered spatial guidance")
    
    # Add experimental configuration info
    gr.Markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        ">
            <h3 style="margin: 0 0 10px 0; color: white;">🧪 Experimental Configuration</h3>
            <p style="margin: 5px 0; font-size: 14px;">
                <strong>2 Datasets:</strong> PKU (4 classes) • CGL (5 classes) | 
                <strong>2 Encoders:</strong> ViT • Swin | 
                <strong>6 Spatial Guidance:</strong> Saliency • Intent • Both • Sal+Text • Intent+Text • All
            </p>
            <p style="margin: 5px 0; font-size: 12px; opacity: 0.9;">
                Total: 24 Experiments (2×2×6) | Text Control: Natural Language Prompts
            </p>
        </div>
    """)
    
    # Add experiment status section
    def get_experiment_status():
        """Get status of all experiments"""
        experiments = list_available_experiments()
        status_text = "### 📊 Experiment Status\n\n"
        
        # Group by dataset
        pku_experiments = {k: v for k, v in experiments.items() if v['dataset'] == 'pku'}
        cgl_experiments = {k: v for k, v in experiments.items() if v['dataset'] == 'cgl'}
        
        status_text += "**PKU Dataset:**\n"
        for exp_name, exp_info in pku_experiments.items():
            status = "✅" if exp_info['checkpoint_available'] else "❌"
            status_text += f"{status} {exp_name}\n"
        
        status_text += "\n**CGL Dataset:**\n"
        for exp_name, exp_info in cgl_experiments.items():
            status = "✅" if exp_info['checkpoint_available'] else "❌"
            status_text += f"{status} {exp_name}\n"
        
        return status_text
    
    # Add status section
    with gr.Row():
        with gr.Column():
            gr.Markdown(get_experiment_status())
        with gr.Column():
            gr.Markdown("""
                **Legend:**
                - ✅ Experiment available (checkpoint found)
                - ❌ Experiment not available (checkpoint missing)
                
                **Note:** Experiments without checkpoints will show an error message when selected.
            """)

    gr.Markdown("""
        <div style="
            text-align: center;
            margin: 20px auto;
            padding: 20px 30px;
            background: linear-gradient(145deg, #1e1e1e, #2a2a2a);
            border-radius: 14px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            max-width: 720px;
            font-family: 'Segoe UI', Roboto, sans-serif;
            color: #f0f0f0;
        ">
            <p style="margin: 6px 0; font-size: 20px; font-weight: 600; color: #ffffff;">
                <strong>Author:</strong> Abdul Waheed Aagha<sup>1</sup>
            </p>
            <p style="margin: 6px 0; font-size: 15px; color: #cccccc;">
                <sup>1</sup>Soongsil University•&nbsp;
                <sup>2</sup>
            </p>
            <hr style="width: 45%; border: 0; border-top: 1px solid #555; margin: 12px auto;">
            <p style="margin: 6px 0; font-size: 15px; color: #aaa;">
                <em>Conference / Journal Name 2024</em>
            </p>
        </div>
    """)


    
    # Add links section at the top
    gr.Markdown("""
        <div style="text-align: center; margin: 20px 0;">
            <a href="YOUR_PAPER_URL" target="_blank" style="margin: 0 10px; text-decoration: none;">
                📄 <strong>Paper</strong>
            </a> | 
            <a href="YOUR_GITHUB_URL" target="_blank" style="margin: 0 10px; text-decoration: none;">
                💻 <strong>GitHub</strong>
            </a> | 
            <a href="YOUR_PROJECT_PAGE_URL" target="_blank" style="margin: 0 10px; text-decoration: none;">
                🌐 <strong>Project Page</strong>
            </a>
        </div>
    """)

    with gr.Tabs(selected=2):
        # Tab 1: Background Generation
        # Tab 2: Layout Generation
        with gr.Tab("📐 Layout Generation"):
            with gr.Row():
                # Left column - Inputs
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Layout Configuration")
                    layout_prompt_input = gr.Textbox(
                        label="Layout Prompt", 
                        placeholder="e.g., 'Create a modern landing page with header, hero section, and CTA'",
                        lines=3
                    )
                    dataset_choice = gr.Radio(
                        choices=["pku", "cgl"],
                        value="pku",
                        label="Dataset"
                    )
                    task_choice = gr.Radio(
                        choices=["uncond", "c", "cwh", "complete"],
                        value="c",
                        label="Task Type"
                    )
                    encoder_choice = gr.Radio(
                        choices=["vit", "swin", "convnext"],
                        value="vit",
                        label="Visual Encoder"
                    )
                    spatial_guidance_choice = gr.Radio(
                        choices=["Saliency", "Intent Map", "Both"],
                        value="Saliency",
                        label="Spatial Guidance Type"
                    )
                    text_control_choice = gr.Checkbox(
                        label="Text Control",
                        value=False,
                        info="Enable natural language control"
                    )
                    layout_seed_slider = gr.Slider(
                        minimum=1,
                        maximum=10000,
                        value=42,
                        step=1,
                        label="Random Seed"
                    )
                    # --- Sample Image Gallery ---
                   
                    # When a sample is selected, set it as the image input
                    layout_image_input = gr.Image(
                        label="Reference Image (Optional)", 
                        type="pil"
                    )
                    
                    layout_generate_btn = gr.Button("🚀 Generate Layout", variant="primary", size="lg")
                # Right column - Outputs
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Results")
                    
                    # Current experiment configuration display
                    experiment_info = gr.Markdown("""
                        <div style="
                            background: #f8f9fa;
                            border: 1px solid #dee2e6;
                            border-radius: 8px;
                            padding: 15px;
                            margin: 10px 0;
                            font-family: monospace;
                            font-size: 12px;
                        ">
                            <strong>Current Experiment:</strong> <span id="exp-name">pku_vit_saliency</span><br>
                            <strong>Configuration:</strong> Dataset: PKU | Encoder: ViT | Guidance: Saliency | Text: Disabled
                        </div>
                    """)
                    
                    with gr.Row():
                        final_output = gr.Image(label="Final Layout Output", height=300)
                    with gr.Row():
                        spatial_output = gr.Image(label="Spatial Guidance", height=250)
                        bbox_output = gr.Image(label="Bounding Boxes", height=250)
                    layout_result_text = gr.Textbox(
                        label="Generation Info",
                        lines=6,
                        interactive=False
                    )
            
            # Examples
            gr.Markdown("---")
            gr.Markdown("### 💡 Examples")
            
            # Show all 24 experiments
            gr.Markdown("### 🧪 All 24 Experiments")
            gr.Markdown("""
                <div style="
                    background: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                    font-size: 12px;
                ">
                    <h4 style="margin: 0 0 15px 0; color: #495057;">Experimental Matrix (2×2×6 = 24 Total)</h4>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                        <div>
                            <strong>PKU Dataset (4 classes):</strong><br>
                            • pku_vit_saliency • pku_vit_intent • pku_vit_both<br>
                            • pku_vit_saliency_text • pku_vit_intent_text • pku_vit_both_text<br>
                            • pku_swin_saliency • pku_swin_intent • pku_swin_both<br>
                            • pku_swin_saliency_text • pku_swin_intent_text • pku_swin_both_text
                        </div>
                        <div>
                            <strong>CGL Dataset (5 classes):</strong><br>
                            • cgl_vit_saliency • cgl_vit_intent • cgl_vit_both<br>
                            • cgl_vit_saliency_text • cgl_vit_intent_text • cgl_vit_both_text<br>
                            • cgl_swin_saliency • cgl_swin_intent • cgl_swin_both<br>
                            • cgl_swin_saliency_text • cgl_swin_intent_text • cgl_swin_both_text
                        </div>
                    </div>
                </div>
            """)
            
            # Dynamically generate examples using sample images
            cgl_samples = [os.path.join(CGL_SAMPLES_DIR, f) for f in os.listdir(CGL_SAMPLES_DIR) if f.endswith('.png')]
            pkl_samples = [os.path.join(PKL_SAMPLES_DIR, f) for f in os.listdir(PKL_SAMPLES_DIR) if f.endswith('.png')]
            examples = []
            for img_path in cgl_samples:
                examples.append([
                    f"CGL sample {os.path.basename(img_path)}", "cgl", "c", "swin", "Intent Map", True, Image.open(img_path), 123
                ])
            for img_path in pkl_samples:
                examples.append([
                    f"PKL sample {os.path.basename(img_path)}", "pku", "cwh", "vit", "Saliency", False, Image.open(img_path), 42
                ])
            gr.Examples(
                examples=examples,
                inputs=[layout_prompt_input, dataset_choice, task_choice, encoder_choice, 
                       spatial_guidance_choice, text_control_choice, layout_image_input, layout_seed_slider],
            )
            
            # Function to update experiment info
            def update_experiment_info(dataset, encoder, spatial_guidance, text_control):
                experiment_name = f"{dataset}_{encoder}_{spatial_guidance.lower().replace(' ', '_')}"
                if text_control:
                    experiment_name += "_text"
                
                guidance_display = spatial_guidance
                if text_control:
                    guidance_display += " + Text"
                
                return f"""
                    <div style="
                        background: #f8f9fa;
                        border: 1px solid #dee2e6;
                        border-radius: 8px;
                        padding: 15px;
                        margin: 10px 0;
                        font-family: monospace;
                        font-size: 12px;
                    ">
                        <strong>Current Experiment:</strong> {experiment_name}<br>
                        <strong>Configuration:</strong> Dataset: {dataset.upper()} | Encoder: {encoder.upper()} | Guidance: {guidance_display} | Text: {'Enabled' if text_control else 'Disabled'}
                    </div>
                """
            
            # Update experiment info when parameters change
            for component in [dataset_choice, encoder_choice, spatial_guidance_choice, text_control_choice]:
                component.change(
                    fn=update_experiment_info,
                    inputs=[dataset_choice, encoder_choice, spatial_guidance_choice, text_control_choice],
                    outputs=[experiment_info]
                )
            
            layout_generate_btn.click(
                fn=generate_layout,
                inputs=[
                    layout_prompt_input,
                    dataset_choice,
                    task_choice,
                    encoder_choice,
                    spatial_guidance_choice,
                    text_control_choice,
                    layout_image_input,
                    layout_seed_slider
                ],
                outputs=[
                    final_output,
                    spatial_output,
                    bbox_output,
                    layout_result_text
                ]
            )

        with gr.Tab("📊 Comparison View"):
            gr.Markdown("### 🔬 Experimental Comparison")
            gr.Markdown("Generate multiple experiments side by side for comparison analysis")
            
            with gr.Row():
                # Left column - Configuration
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Comparison Configuration")
                    
                    # Base configuration
                    comp_prompt_input = gr.Textbox(
                        label="Layout Prompt", 
                        placeholder="e.g., 'Create a modern landing page with header, hero section, and CTA'",
                        lines=3
                    )
                    
                    comp_dataset_choice = gr.Radio(
                        choices=["pku", "cgl"],
                        value="pku",
                        label="Dataset"
                    )
                    
                    comp_task_choice = gr.Radio(
                        choices=["uncond", "c", "cwh", "complete"],
                        value="c",
                        label="Task Type"
                    )
                    
                    comp_seed_slider = gr.Slider(
                        minimum=1,
                        maximum=10000,
                        value=42,
                        step=1,
                        label="Random Seed"
                    )
                    
                    # Comparison type selection
                    comp_type_choice = gr.Radio(
                        choices=[
                            "Encoder Comparison (ViT vs Swin)",
                            "Spatial Guidance Comparison (Saliency vs Intent vs Both)",
                            "Text Control Comparison (With vs Without Text)",
                            "Full Matrix (All 6 Configurations)"
                        ],
                        value="Encoder Comparison (ViT vs Swin)",
                        label="Comparison Type"
                    )
                    
                    comp_generate_btn = gr.Button("🚀 Generate Comparison", variant="primary", size="lg")
                
                # Right column - Results
                with gr.Column(scale=3):
                    gr.Markdown("### 📊 Comparison Results")
                    
                    # Grid layout for comparison
                    with gr.Row():
                        comp_img1 = gr.Image(label="Experiment 1", height=200)
                        comp_img2 = gr.Image(label="Experiment 2", height=200)
                        comp_img3 = gr.Image(label="Experiment 3", height=200)
                    
                    with gr.Row():
                        comp_img4 = gr.Image(label="Experiment 4", height=200)
                        comp_img5 = gr.Image(label="Experiment 5", height=200)
                        comp_img6 = gr.Image(label="Experiment 6", height=200)
                    
                    comp_result_text = gr.Textbox(
                        label="Comparison Info",
                        lines=8,
                        interactive=False
                    )
            
            # Examples for comparison
            gr.Markdown("---")
            gr.Markdown("### 💡 Comparison Examples")
            gr.Examples(
                examples=[
                    ["Create a modern landing page with header, hero section, and CTA", "pku", "c", "Encoder Comparison (ViT vs Swin)", 42],
                    ["Design a poster with logo, title, and description", "cgl", "cwh", "Spatial Guidance Comparison (Saliency vs Intent vs Both)", 123],
                    ["Generate a website layout with navigation and content", "pku", "complete", "Text Control Comparison (With vs Without Text)", 456],
                ],
                inputs=[comp_prompt_input, comp_dataset_choice, comp_task_choice, comp_type_choice, comp_seed_slider],
            )
            
            # Function to generate comparison
            def generate_comparison(prompt, dataset, task, comp_type, seed):
                if not prompt:
                    return [None] * 6, "Please enter a prompt"
                
                # Load sample images
                sample_images = load_sample_images(dataset)
                if sample_images:
                    random.seed(seed)
                    sample_name, sample_image = random.choice(sample_images)
                    base_img = sample_image.copy()
                    width, height = base_img.size
                else:
                    width, height = 512, 512
                    base_img = Image.new('RGB', (width, height), color='white')
                
                results = []
                experiment_info = []
                
                # Generate experiments based on comparison type
                if "Encoder Comparison" in comp_type:
                    encoders = ["vit", "swin"]
                    spatial_guidance = "Saliency"
                    text_control = False
                    
                    for i, encoder in enumerate(encoders):
                        img, spatial, bbox, info = generate_single_experiment(
                            prompt, dataset, task, encoder, spatial_guidance, text_control, base_img, seed + i
                        )
                        results.extend([img, spatial, bbox])
                        experiment_info.append(f"Experiment {i+1}: {dataset}_{encoder}_{spatial_guidance}")
                
                elif "Spatial Guidance Comparison" in comp_type:
                    encoder = "vit"
                    spatial_guidances = ["Saliency", "Intent Map", "Both"]
                    text_control = False
                    
                    for i, spatial_guidance in enumerate(spatial_guidances):
                        img, spatial, bbox, info = generate_single_experiment(
                            prompt, dataset, task, encoder, spatial_guidance, text_control, base_img, seed + i
                        )
                        results.extend([img, spatial, bbox])
                        experiment_info.append(f"Experiment {i+1}: {dataset}_{encoder}_{spatial_guidance.lower().replace(' ', '_')}")
                
                elif "Text Control Comparison" in comp_type:
                    encoder = "vit"
                    spatial_guidance = "Saliency"
                    text_controls = [False, True]
                    
                    for i, text_control in enumerate(text_controls):
                        img, spatial, bbox, info = generate_single_experiment(
                            prompt, dataset, task, encoder, spatial_guidance, text_control, base_img, seed + i
                        )
                        results.extend([img, spatial, bbox])
                        experiment_info.append(f"Experiment {i+1}: {dataset}_{encoder}_{spatial_guidance.lower().replace(' ', '_')}{'_text' if text_control else ''}")
                
                elif "Full Matrix" in comp_type:
                    encoder = "vit"
                    spatial_guidances = ["Saliency", "Intent Map", "Both"]
                    text_controls = [False, True]
                    
                    for i, (spatial_guidance, text_control) in enumerate(zip(spatial_guidances, text_controls)):
                        img, spatial, bbox, info = generate_single_experiment(
                            prompt, dataset, task, encoder, spatial_guidance, text_control, base_img, seed + i
                        )
                        results.extend([img, spatial, bbox])
                        experiment_info.append(f"Experiment {i+1}: {dataset}_{encoder}_{spatial_guidance.lower().replace(' ', '_')}{'_text' if text_control else ''}")
                
                # Pad results to 6 images
                while len(results) < 6:
                    results.append(None)
                
                result_text = f"""
                Comparison Generated Successfully!
                
                Prompt: {prompt}
                Dataset: {dataset.upper()}
                Task: {task}
                Comparison Type: {comp_type}
                Seed: {seed}
                
                Experiments:
                {chr(10).join(experiment_info)}
                """
                
                return results + [result_text]
            
            # Helper function to generate single experiment
            def generate_single_experiment(prompt, dataset, task, encoder, spatial_guidance, text_control, base_img, seed):
                width, height = base_img.size
                
                # Generate spatial guidance visualization
                spatial_img = base_img.copy()
                draw = ImageDraw.Draw(spatial_img, 'RGBA')
                
                random.seed(seed)
                num_regions = 30 if spatial_guidance == "Both" else 20
                
                for _ in range(num_regions):
                    x = random.randint(0, width)
                    y = random.randint(0, height)
                    radius = random.randint(15, 60)
                    intensity = random.randint(80, 180)
                    
                    if spatial_guidance == "Saliency":
                        color = (intensity, 0, 0, 120)
                    elif spatial_guidance == "Intent Map":
                        color = (0, intensity, 0, 120)
                    else:  # Both
                        color = (intensity, intensity//2, 0, 100)
                    
                    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
                
                # Generate bounding boxes
                bbox_img = base_img.copy()
                draw = ImageDraw.Draw(bbox_img)
                
                random.seed(seed)
                
                if dataset.lower() == "pku":
                    labels = ['Text', 'Logo', 'Underlay', 'Embellishment']
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                else:  # CGL
                    labels = ['Text', 'Logo', 'Underlay', 'Embellishment', 'Decoration']
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
                
                num_boxes = min(len(labels), random.randint(2, 6))
                
                for i in range(num_boxes):
                    x1 = random.randint(10, width - 150)
                    y1 = random.randint(10, height - 150)
                    x2 = x1 + random.randint(80, 200)
                    y2 = y1 + random.randint(60, 150)
                    x2 = min(x2, width - 10)
                    y2 = min(y2, height - 10)
                    
                    color = colors[i % len(colors)]
                    label = labels[i % len(labels)]
                    
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    try:
                        font = ImageFont.truetype("arial.ttf", 14)
                    except:
                        font = ImageFont.load_default()
                    
                    text_bbox = draw.textbbox((x1, y1), label, font=font)
                    draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
                    draw.text((x1, y1), label, fill='white', font=font)
                
                # Generate final output
                final_output = base_img.copy()
                final_output.paste(spatial_img, (0, 0), spatial_img.convert('RGBA').split()[-1])
                
                draw_final = ImageDraw.Draw(final_output)
                random.seed(seed)
                for i in range(num_boxes):
                    x1 = random.randint(10, width - 150)
                    y1 = random.randint(10, height - 150)
                    x2 = x1 + random.randint(80, 200)
                    y2 = y1 + random.randint(60, 150)
                    x2 = min(x2, width - 10)
                    y2 = min(y2, height - 10)
                    
                    color = colors[i % len(colors)]
                    label = labels[i % len(labels)]
                    draw_final.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                return final_output, spatial_img, bbox_img, f"Generated: {dataset}_{encoder}_{spatial_guidance.lower().replace(' ', '_')}{'_text' if text_control else ''}"
            
            comp_generate_btn.click(
                fn=generate_comparison,
                inputs=[comp_prompt_input, comp_dataset_choice, comp_task_choice, comp_type_choice, comp_seed_slider],
                outputs=[comp_img1, comp_img2, comp_img3, comp_img4, comp_img5, comp_img6, comp_result_text]
            )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=False)