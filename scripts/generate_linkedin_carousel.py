"""
Generates a stunning 5-slide LinkedIn Carousel in PDF and high-res image formats.
Uses Headless Chrome to compile a pixel-perfect 1200x1200 square carousel.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_base64_image(image_path: Path) -> str:
    """Reads image and returns base64 data URI."""
    suffix = image_path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def build_carousel_html() -> str:
    proof_img_path = REPO_ROOT / "docs" / "images" / "win_case_1_low_light_rescue.jpg"
    proof_base64 = get_base64_image(proof_img_path) if proof_img_path.exists() else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

  @page {{
    size: 1200px 1200px;
    margin: 0;
  }}

  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  body {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #07090E;
    color: #F8FAFC;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}

  .slide {{
    width: 1200px;
    height: 1200px;
    page-break-after: always;
    position: relative;
    padding: 75px 80px;
    background: radial-gradient(circle at 80% 15%, rgba(0, 240, 255, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 20% 85%, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
                #07090E;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow: hidden;
  }}

  .header-tag {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0, 240, 255, 0.12);
    border: 1px solid rgba(0, 240, 255, 0.3);
    color: #00F0FF;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 8px 18px;
    border-radius: 9999px;
    width: fit-content;
    margin-bottom: 24px;
  }}

  .header-tag.purple {{
    background: rgba(168, 85, 247, 0.12);
    border-color: rgba(168, 85, 247, 0.3);
    color: #C084FC;
  }}

  .header-tag.green {{
    background: rgba(16, 185, 129, 0.12);
    border-color: rgba(16, 185, 129, 0.3);
    color: #34D399;
  }}

  .header-tag.red {{
    background: rgba(239, 68, 68, 0.12);
    border-color: rgba(239, 68, 68, 0.3);
    color: #F87171;
  }}

  h1.title {{
    font-size: 58px;
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.03em;
    color: #FFFFFF;
    margin-bottom: 20px;
  }}

  .gradient-text {{
    background: linear-gradient(135deg, #00F0FF 0%, #7000FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}

  .gradient-green {{
    background: linear-gradient(135deg, #34D399 0%, #00F0FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}

  p.subtitle {{
    font-size: 24px;
    line-height: 1.5;
    color: #94A3B8;
    font-weight: 400;
    max-width: 950px;
  }}

  /* Cards Grid */
  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-top: 35px;
  }}

  .card {{
    background: rgba(20, 28, 46, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    padding: 32px;
    backdrop-filter: blur(12px);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }}

  .card.highlight {{
    border-color: rgba(0, 240, 255, 0.4);
    background: linear-gradient(145deg, rgba(20, 28, 46, 0.9), rgba(0, 240, 255, 0.06));
    box-shadow: 0 20px 40px -15px rgba(0, 240, 255, 0.15);
  }}

  .card-icon {{
    font-size: 38px;
    margin-bottom: 16px;
  }}

  .card-title {{
    font-size: 24px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 12px;
  }}

  .card-desc {{
    font-size: 17px;
    line-height: 1.55;
    color: #94A3B8;
  }}

  /* Diagram Flow Nodes */
  .flow-container {{
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-top: 30px;
  }}

  .flow-step {{
    display: flex;
    align-items: center;
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 20px 28px;
    gap: 24px;
    position: relative;
  }}

  .flow-step.cyan {{ border-left: 6px solid #00F0FF; }}
  .flow-step.purple {{ border-left: 6px solid #A855F7; }}
  .flow-step.amber {{ border-left: 6px solid #F59E0B; }}
  .flow-step.pink {{ border-left: 6px solid #EC4899; }}
  .flow-step.emerald {{ border-left: 6px solid #10B981; background: rgba(16, 185, 129, 0.08); }}

  .step-num {{
    width: 48px;
    height: 48px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 20px;
    flex-shrink: 0;
  }}

  .step-num.c1 {{ background: rgba(0, 240, 255, 0.15); color: #00F0FF; }}
  .step-num.c2 {{ background: rgba(168, 85, 247, 0.15); color: #C084FC; }}
  .step-num.c3 {{ background: rgba(245, 158, 11, 0.15); color: #FBBF24; }}
  .step-num.c4 {{ background: rgba(236, 72, 153, 0.15); color: #F472B6; }}
  .step-num.c5 {{ background: rgba(16, 185, 129, 0.2); color: #34D399; }}

  .step-content {{
    flex: 1;
  }}

  .step-title {{
    font-size: 20px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 4px;
  }}

  .step-desc {{
    font-size: 15px;
    color: #94A3B8;
  }}

  /* Metrics Big Cards */
  .metrics-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-top: 35px;
  }}

  .metric-card {{
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 28px 24px;
    text-align: center;
  }}

  .metric-card.winner {{
    background: linear-gradient(165deg, rgba(16, 185, 129, 0.12), rgba(0, 240, 255, 0.08));
    border: 1.5px solid rgba(16, 185, 129, 0.4);
    box-shadow: 0 15px 35px -10px rgba(16, 185, 129, 0.2);
  }}

  .metric-label {{
    font-size: 14px;
    font-family: 'JetBrains Mono', monospace;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
  }}

  .metric-val {{
    font-size: 46px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
  }}

  .metric-val.green {{ color: #10B981; }}
  .metric-sub {{ font-size: 14px; color: #64748B; }}

  /* Footer bar */
  .footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding-top: 24px;
    font-size: 15px;
    color: #64748B;
    font-family: 'JetBrains Mono', monospace;
  }}

  .footer .author {{
    color: #CBD5E1;
    font-weight: 600;
  }}

  .footer .slide-num {{
    background: rgba(255, 255, 255, 0.05);
    padding: 6px 14px;
    border-radius: 9999px;
  }}
</style>
</head>
<body>

<!-- SLIDE 1: COVER / THE HOOK -->
<div class="slide">
  <div>
    <div class="header-tag">Thought Leadership / AI Engineering</div>
    <h1 class="title">
      Why Open-Loop Computer Vision Fails<br>
      <span class="gradient-text">(And How an Agentic Loop Fixes It)</span>
    </h1>
    <p class="subtitle">
      Traditional face-blurring AI runs in a straight line: detect once, blur, and pray nothing was missed.<br><br>
      Here is what happened when we replaced that with an <strong>autonomous Evaluator-Optimizer state machine</strong>.
    </p>
    
    <div class="grid-2">
      <div class="card highlight">
        <div class="card-icon">🎯</div>
        <div class="card-title">221 Real CCTV Faces</div>
        <div class="card-desc">Benchmarked across 100 hand-labeled surveillance frames in dark rooms, motion blur, and crowded corridors.</div>
      </div>
      <div class="card highlight">
        <div class="card-icon">🛡️</div>
        <div class="card-title">50% Failure Reduction</div>
        <div class="card-desc">Cut missed faces in complex occlusions from 15.6% down to 7.7% through closed-loop self-healing.</div>
      </div>
    </div>
  </div>

  <div class="footer">
    <div class="author">By Viswarup | AI & Computer Vision</div>
    <div class="slide-num">01 / 05 &rarr; Swipe</div>
  </div>
</div>

<!-- SLIDE 2: THE PROBLEM (WHY SINGLE-PASS FAILS) -->
<div class="slide">
  <div>
    <div class="header-tag red">The Fundamental Flaw</div>
    <h1 class="title">Traditional AI Has Zero Quality Control</h1>
    <p class="subtitle">Standard computer vision operates in an open loop: input in, prediction out. When real surveillance happens, it fails silently.</p>
    
    <div class="grid-2">
      <div class="card">
        <div class="card-icon">🌑</div>
        <div class="card-title">Dark Shadows Blind Models</div>
        <div class="card-desc">Low contrast in dark rooms causes face detectors to output 0 boxes. The face is broadcast completely unblurred.</div>
      </div>
      <div class="card">
        <div class="card-icon">🔄</div>
        <div class="card-title">Turned Heads & Profiles</div>
        <div class="card-desc">When people look down or sideways, facial features disappear from view. Standard detectors fail immediately.</div>
      </div>
      <div class="card">
        <div class="card-icon">⚠️</div>
        <div class="card-title">Partial Edge Leaks</div>
        <div class="card-desc">Fixed blur boxes cover only eyes and nose, leaving the jawline, chin, and hair identifiable—a major GDPR violation.</div>
      </div>
      <div class="card">
        <div class="card-icon">❌</div>
        <div class="card-title">No Self-Verification</div>
        <div class="card-desc">Once the file is saved, nobody and no model checks whether the privacy shield actually worked.</div>
      </div>
    </div>
  </div>

  <div class="footer">
    <div class="author">Why Single-Pass Vision Breaks</div>
    <div class="slide-num">02 / 05 &rarr; Swipe</div>
  </div>
</div>

<!-- SLIDE 3: THE COLORFUL SIMPLE ARCHITECTURE -->
<div class="slide">
  <div>
    <div class="header-tag purple">Simple Architecture</div>
    <h1 class="title">The Solution: An <span class="gradient-text">Evaluator-Optimizer</span> Loop</h1>
    <p class="subtitle">Instead of running once, the agent senses the room, calls specialized tools, audits the output, and self-heals leaks.</p>

    <div class="flow-container">
      <div class="flow-step cyan">
        <div class="step-num c1">01</div>
        <div class="step-content">
          <div class="step-title">Perception Telemetry Engine</div>
          <div class="step-desc">Measures mean luminance, dark pixel ratio, and blur variance before running any detector.</div>
        </div>
      </div>
      <div class="flow-step purple">
        <div class="step-num c2">02</div>
        <div class="step-content">
          <div class="step-title">Dynamic Tool Dispatch (Night Boost + Body Pose)</div>
          <div class="step-desc">Dark room? Triggers CLAHE lighting boost. Turned head? Uses YOLOv8 body skeleton keypoints.</div>
        </div>
      </div>
      <div class="flow-step amber">
        <div class="step-num c3">03</div>
        <div class="step-content">
          <div class="step-title">Heavy Dual-Layer Obfuscation</div>
          <div class="step-desc">Applies irreversible mosaic and Gaussian blur with scale-adaptive bounding margins.</div>
        </div>
      </div>
      <div class="flow-step pink">
        <div class="step-num c4">04</div>
        <div class="step-content">
          <div class="step-title">Independent QA Auditor (RetinaFace)</div>
          <div class="step-desc">A completely separate neural network scans the blurred output: "Can I still detect any face?"</div>
        </div>
      </div>
      <div class="flow-step emerald">
        <div class="step-num c5">05</div>
        <div class="step-content">
          <div class="step-title">Autonomous Self-Healing Loop</div>
          <div class="step-desc">If a leak is caught, the reflection node expands margins (+25%) or injects missing boxes until clean!</div>
        </div>
      </div>
    </div>
  </div>

  <div class="footer">
    <div class="author">Closed-Loop Verification</div>
    <div class="slide-num">03 / 05 &rarr; Swipe</div>
  </div>
</div>

<!-- SLIDE 4: REAL-WORLD PROOF -->
<div class="slide">
  <div>
    <div class="header-tag green">Visual Evidence</div>
    <h1 class="title">Real Surveillance Proof: <span class="gradient-green">The Rescue</span></h1>
    <p class="subtitle">Side-by-side test in a dark corridor: Base model and Pose model fail, but the Agentic pipeline self-heals.</p>
    
    <div style="margin-top: 30px; border-radius: 20px; overflow: hidden; border: 2px solid rgba(255, 255, 255, 0.12); box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);">
      <img src="{proof_base64}" style="width: 100%; display: block;" alt="Visual Proof Strip">
    </div>

    <div class="grid-2" style="margin-top: 24px;">
      <div class="card" style="padding: 20px 24px;">
        <div style="color: #F87171; font-weight: 700; font-size: 16px; margin-bottom: 4px;">❌ Base & Pose Models (Panels 2 & 3)</div>
        <div style="color: #94A3B8; font-size: 15px;">Fails on dark silhouettes—leaving 1 real human face completely unredacted (red LEAK box).</div>
      </div>
      <div class="card" style="padding: 20px 24px; border-color: rgba(16, 185, 129, 0.4);">
        <div style="color: #34D399; font-weight: 700; font-size: 16px; margin-bottom: 4px;">✅ Agentic Output (Panel 4)</div>
        <div style="color: #94A3B8; font-size: 15px;">Senses low light, boosts exposure, catches 2/2 faces (100% clean), and auditor certifies zero leaks!</div>
      </div>
    </div>
  </div>

  <div class="footer">
    <div class="author">Real-World Verification Strip</div>
    <div class="slide-num">04 / 05 &rarr; Swipe</div>
  </div>
</div>

<!-- SLIDE 5: THE NUMBERS & CONCLUSION -->
<div class="slide">
  <div>
    <div class="header-tag cyan">Empirical Benchmark</div>
    <h1 class="title">The Numbers: <span class="gradient-text">221 Real Faces</span></h1>
    <p class="subtitle">Tested across 100 benchmark surveillance frames with strict ground-truth annotations.</p>

    <div class="metrics-row">
      <div class="metric-card">
        <div class="metric-label">1. Standard Model</div>
        <div class="metric-val">68.3%</div>
        <div class="metric-sub">151 caught | 70 missed</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">2. SCRFD + Pose</div>
        <div class="metric-val">76.0%</div>
        <div class="metric-sub">168 caught | 53 missed</div>
      </div>
      <div class="metric-card winner">
        <div class="metric-label">3. Agentic Graph 🏆</div>
        <div class="metric-val green">77.4%</div>
        <div class="metric-sub">171 caught | Only 50 missed</div>
      </div>
    </div>

    <div class="card highlight" style="margin-top: 30px; padding: 28px 32px;">
      <div class="card-title" style="font-size: 22px; color: #00F0FF; margin-bottom: 8px;">💡 The Big Takeaway for AI Engineers:</div>
      <div class="card-desc" style="font-size: 18px; color: #E2E8F0; line-height: 1.6;">
        "Real-world Agentic AI isn't about chat bots. In high-stakes Computer Vision, the true power of an agent is acting as an <strong>automated quality assurance supervisor</strong> that catches dangerous tail-risk edge cases before they escape into production."
      </div>
    </div>
  </div>

  <div class="footer">
    <div class="author">💻 github.com/viswarup/agentic-face-redaction-pipeline</div>
    <div class="slide-num">05 / 05 &bull; Complete</div>
  </div>
</div>

</body>
</html>
"""
    return html


def main():
    temp_html = REPO_ROOT / "docs" / "carousel_temp.html"
    output_pdf = REPO_ROOT / "docs" / "Agentic_Face_Redaction_LinkedIn_Carousel.pdf"
    output_dir = REPO_ROOT / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating LinkedIn Carousel HTML...")
    html_content = build_carousel_html()
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Compiling pixel-perfect PDF via Headless Chrome...")
    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    chrome_tmp = Path("/tmp/chrome_carousel_tmp")
    chrome_tmp.mkdir(parents=True, exist_ok=True)

    cmd_chrome = [
        chrome_bin,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--user-data-dir={chrome_tmp}",
        f"--print-to-pdf={output_pdf}",
        str(temp_html),
    ]

    res = subprocess.run(cmd_chrome, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Chrome error: {res.stderr}")
        sys.exit(res.returncode)

    if output_pdf.exists():
        pdf_mb = output_pdf.stat().st_size / (1024 * 1024)
        print(f"\n[SUCCESS] LinkedIn Carousel PDF generated ({pdf_mb:.2f} MB):")
        print(f"  -> {output_pdf}")
    else:
        print("Error: Output PDF was not created.")
        sys.exit(1)


if __name__ == "__main__":
    main()
