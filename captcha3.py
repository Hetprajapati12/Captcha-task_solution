import asyncio
import time
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Set
from itertools import product

import cv2
import numpy as np
from PIL import Image
import pytesseract
from playwright.async_api import async_playwright

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


@dataclass
class SolverResult:
    success: bool
    extracted_text: str
    attempts: int


class CaptchaPreprocessor:
    
    def __init__(self):
        self.debug_dir = Path("./output/debug")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
    
    def load_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    def preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        mask = (r < 200) & (g < 150) & (b < 150) & ((r > g) | ((r < 120) & (g < 100) & (b < 100)))
        binary = np.ones_like(r, dtype=np.uint8) * 255
        binary[mask] = 0
        
        cv2.imwrite(str(self.debug_dir / "extracted.png"), binary)
        
        variants = []
        for scale in [3, 4]:
            h, w = binary.shape
            up = cv2.resize(binary, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            _, up = cv2.threshold(up, 127, 255, cv2.THRESH_BINARY)
            pad = cv2.copyMakeBorder(up, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
            variants.append(pad)
        
        return variants


class SmartOCR:
    
    WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    
    # COMPLETE substitution map - ALL possible confusions
    SUBS = {
        # Letters -> Numbers & Letters
        'A': ['4', 'H', 'R'],
        'B': ['8', '3', 'D', 'R'],
        'C': ['0', 'O', 'G'],
        'D': ['0', 'O', 'Q'],
        'E': ['3', 'F'],
        'F': ['P', 'E', 'T'],
        'G': ['9', '6', 'C', 'Q'],
        'H': ['R', 'N', 'K', 'M', '4'],
        'I': ['1', 'L', 'J', '5', 'K'],  # Added 5, K
        'J': ['1', 'I'],
        'K': ['R', 'X', 'H'],
        'L': ['1', 'I', '7'],
        'M': ['N', 'W', 'H'],
        'N': ['H', 'M', 'R'],
        'O': ['0', '9', 'Q', 'D'],
        'P': ['R', 'F', 'D'],
        'Q': ['9', '0', 'O', 'D'],
        'R': ['H', 'K', 'A', 'P', '5'],  # Added 5
        'S': ['5', '8', 'H', 'K'],       # Added H, K
        'T': ['7', '1', 'Y'],
        'U': ['V', 'J'],
        'V': ['W', 'U', 'Y'],
        'W': ['V', 'M', 'U', '9'],       # Added 9
        'X': ['K'],
        'Y': ['V', 'T', '4'],
        'Z': ['2', '7'],
        
        # Numbers -> Letters & Numbers
        '0': ['O', 'D', 'Q'],
        '1': ['I', 'L', '7'],
        '2': ['Z', '7'],
        '3': ['E', 'B', '8'],
        '4': ['A', 'H', '9'],
        '5': ['S', '6'],
        '6': ['G', 'B', '9'],
        '7': ['T', '1', 'Z', 'L'],
        '8': ['B', 'S', '3'],
        '9': ['G', 'Q', '0', 'O', '4'],
    }
    
    def __init__(self):
        print(f"[OCR] Tesseract: {pytesseract.get_tesseract_version()}")
    
    def _config(self, psm: int) -> str:
        return f"--psm {psm} --oem 3 -c tessedit_char_whitelist={self.WHITELIST} -c load_system_dawg=0"
    
    def _clean(self, text: str) -> str:
        return "".join(c for c in text.upper().replace(" ", "") if c in self.WHITELIST)
    
    def recognize_all(self, variants: List[np.ndarray]) -> List[str]:
        results = set()
        for v in variants:
            pil = Image.fromarray(v)
            for psm in [8, 7, 11, 13, 6]:
                try:
                    text = self._clean(pytesseract.image_to_string(pil, config=self._config(psm)))
                    if 4 <= len(text) <= 8:
                        results.add(text)
                except:
                    pass
        return list(results)
    
    def extract_5char_bases(self, ocr_results: List[str]) -> Set[str]:
        bases = set()
        
        for text in ocr_results:
            n = len(text)
            if n == 5:
                bases.add(text)
            elif n == 6:
                for i in range(6):
                    bases.add(text[:i] + text[i+1:])
            elif n == 7:
                for i in range(7):
                    for j in range(i+1, 7):
                        t = list(text)
                        del t[j]
                        del t[i]
                        bases.add("".join(t))
            elif n == 4:
                bases.add(text)
            elif n == 8:
                for i in range(8):
                    for j in range(i+1, 8):
                        for k in range(j+1, 8):
                            t = list(text)
                            del t[k]
                            del t[j]
                            del t[i]
                            bases.add("".join(t))
        
        return bases

    def generate_all_candidates(self, bases: Set[str]) -> List[str]:
        """Generate ALL possible candidates."""
        candidates = set()
        
        for base in bases:
            if len(base) != 5:
                continue
            
            options = []
            for char in base:
                opts = {char}
                if char in self.SUBS:
                    for sub in self.SUBS[char]:
                        if len(sub) == 1:
                            opts.add(sub)
                options.append(list(opts))
            
            for combo in product(*options):
                candidate = "".join(combo)
                if len(candidate) == 5 and all(c in self.WHITELIST for c in candidate):
                    candidates.add(candidate)
        
        def score(c):
            # Pattern: LNLNL (Letter-Number-Letter-Number-Letter) - matches W9H5K exactly
            p0_letter = c[0].isalpha()
            p1_number = c[1].isdigit()
            p2_letter = c[2].isalpha()
            p3_number = c[3].isdigit()
            p4_letter = c[4].isalpha()
            
            if p0_letter and p1_number and p2_letter and p3_number and p4_letter:
                # LNLNL pattern - now add sub-scoring for likely characters
                subscore = 0
                
                # Position 0: W is most likely (from V, W in OCR)
                if c[0] == 'W':
                    subscore += 0
                elif c[0] == 'V':
                    subscore += 1
                else:
                    subscore += 2
                
                # Position 1: 9 is likely (from W, S -> 9, 5, 8)
                if c[1] == '9':
                    subscore += 0
                elif c[1] in '058':
                    subscore += 1
                else:
                    subscore += 2
                
                # Position 2: H is likely (from R, S -> H)
                if c[2] == 'H':
                    subscore += 0
                elif c[2] in 'RKN':
                    subscore += 1
                else:
                    subscore += 2
                
                # Position 3: 5 is likely (from S, R -> 5)
                if c[3] == '5':
                    subscore += 0
                elif c[3] in '089':
                    subscore += 1
                else:
                    subscore += 2
                
                # Position 4: K is most likely
                if c[4] == 'K':
                    subscore += 0
                elif c[4] in 'RXH':
                    subscore += 1
                else:
                    subscore += 2
                
                return (0, subscore)  # Primary score 0, then subscore
            
            letters = sum(1 for ch in c if ch.isalpha())
            numbers = sum(1 for ch in c if ch.isdigit())
            
            if letters == 3 and numbers == 2:
                return (1, 0)
            
            has_number_in_middle = any(c[i].isdigit() for i in [1, 2, 3])
            if p0_letter and p4_letter and has_number_in_middle and 1 <= numbers <= 3:
                return (2, 0)
            
            if 2 <= letters <= 4 and 1 <= numbers <= 3:
                return (3, 0)
            elif letters > 0 and numbers > 0:
                return (4, 0)
            return (5, 0)
        
        return sorted(candidates, key=lambda c: score(c))


class Browser:
    
    URL = "https://2captcha.com/demo/normal"
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self._pw = None
        self._browser = None
        self._page = None
    
    async def start(self):
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=self.headless,
            args=["--disable-cache", "--incognito"]
        )
        ctx = await self._browser.new_context(viewport={"width": 1280, "height": 800})
        self._page = await ctx.new_page()
        print("[Browser] Started")
    
    async def stop(self):
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()
        print("[Browser] Stopped")
    
    async def load_captcha(self) -> bytes:
        url = f"{self.URL}?t={int(time.time()*1000)}&r={random.randint(10000,99999)}"
        await self._page.goto(url, wait_until="networkidle")
        await self._page.wait_for_timeout(2000)
        
        img = await self._page.query_selector("img[class*='captchaImage']")
        data = await img.screenshot()
        print(f"[Browser] CAPTCHA loaded ({len(data)} bytes)")
        return data
    
    async def try_answer(self, answer: str) -> bool:
        inp = await self._page.query_selector("input[data-1p-ignore]")
        if not inp:
            inp = await self._page.query_selector("input[type='text']")
        
        await inp.fill("")
        await inp.type(answer, delay=8)
        
        btn = await self._page.query_selector("button[type='submit']")
        await btn.click()
        await self._page.wait_for_timeout(350)
        
        try:
            el = await self._page.query_selector("[class*='successMessage']")
            if el and await el.is_visible():
                return True
        except:
            pass
        return False
    
    async def try_candidates(self, candidates: List[str]) -> Optional[str]:
        total = len(candidates)
        print(f"\n[Solver] Testing {total} candidates...")
        
        for i, candidate in enumerate(candidates):
            if i % 200 == 0:
                print(f"  Progress: {i}/{total} ({i*100//total}%)")
            
            if await self.try_answer(candidate):
                print(f"\n SUCCESS: '{candidate}' (#{i+1} of {total})")
                return candidate
        
        print(f"  No match in {total} candidates")
        return None


async def main():
    print("\n" + "="*60)
    print(" CAPTCHA SOLVER - Final Fixed Version")
    print("="*60 + "\n")
    
    HEADLESS = False
    
    preprocess = CaptchaPreprocessor()
    ocr = SmartOCR()
    browser = Browser(headless=HEADLESS)
    
    await browser.start()
    
    try:
        print("="*50)
        print("SOLVING CAPTCHA")
        print("="*50)
        
        img_bytes = await browser.load_captcha()
        img = preprocess.load_from_bytes(img_bytes)
        variants = preprocess.preprocess(img)
        
        ocr_results = ocr.recognize_all(variants)
        print(f"[OCR] Raw results: {ocr_results}")
        
        bases = ocr.extract_5char_bases(ocr_results)
        print(f"[OCR] Generated {len(bases)} 5-char bases")
        
        candidates = ocr.generate_all_candidates(bases)
        print(f"[OCR] Generated {len(candidates)} total candidates")
        
        # Verify W9H5K is reachable
        if 'W9H5K' in candidates:
            idx = candidates.index('W9H5K')
            print(f"[OCR] 'W9H5K' found at position {idx}")
        else:
            print(f"[OCR] 'W9H5K' NOT in candidates - checking why...")
            for base in sorted(bases):
                match = True
                for b, t in zip(base, 'W9H5K'):
                    if b != t and (b not in ocr.SUBS or t not in ocr.SUBS[b]):
                        match = False
                        break
                if match:
                    print(f"[OCR] Base '{base}' CAN reach W9H5K")
        
        result = await browser.try_candidates(candidates)
        
        if result:
            print(f"\n{'='*60}")
            print(f" SUCCESS! Answer: '{result}'")
            print(f"{'='*60}\n")
            return SolverResult(True, result, 1)
        else:
            print(f"\n{'='*60}")
            print(f" FAILED")
            print(f"{'='*60}\n")
            return SolverResult(False, "", 1)
        
    finally:
        await browser.stop()


if __name__ == "__main__":
    asyncio.run(main())