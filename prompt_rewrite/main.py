from fastapi import FastAPI, Depends
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import asyncio
import asyncpg 
from openrouter import OpenRouter

load_dotenv()
PSTG_URL = os.getenv("PSTG_URL")
OPEN_ROUTER = os.getenv("OPEN_ROUTER")


db_pool = None

class Payload(BaseModel):
    action: str
    companyName: str
    userId: str
    timestamp: str
    feedbackId: str
    instruction: str
    aiMessage: str
    userMessage: str
    comment: str
    importance: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool

    db_pool = await asyncpg.create_pool(PSTG_URL)
    
    yield

    await db_pool.close()

app = FastAPI(lifespan=lifespan)

async def get_connection():
    async with db_pool.acquire() as connection:
        yield connection

@app.post("/feedback")
async def main(data: Payload, connection = Depends(get_connection)):

    while True:
        row = await connection.fetchrow('''
        UPDATE ai_persona_config
        SET is_locked = true,
            locked_at = NOW()
        WHERE company_name = $1
          AND (is_locked = false OR locked_at < NOW() - INTERVAL '5 minutes')
        RETURNING is_locked, locked_at
    ''', data.companyName)

        if row:
            break
        
        await asyncio.sleep(5)
    try:
        row = await connection.fetchrow('''
            SELECT learned_instructions, hard_rules
            FROM ai_persona_config
            WHERE company_name = $1
        ''', data.companyName)

        user_input = f'''
        **Feedback szefa firmy:** 
        Wiadomość klienta: {data.userMessage}
        Odpowiedź chatbota: {data.aiMessage}
        Feedback szefa firmy na odpowiedź chatbota: {data.comment}
        Poziom ważności(1-3): {data.importance}
        '''
        
        if data.action == "delete":
            service_prompt = f'''
            ### ROLA
            Jesteś Architektem Zachowań AI (AI Behavior Architect). Jesteś absolutnie genialnym prompt engineerem Twoim zadaniem jest usuwanie z listy "Wyuczonych Instrukcji" (learned_instructions) niechcianych zasad dla chatbota biznesowego. Nie jesteś chatbotem – jesteś inżynierem, który go programuje.

            ### CEL
            Otrzymasz informację zwrotną (Feedback) od szefa dotyczącą konkretnej odpowiedzi chatbota. Otrzymasz wiadomość klienta do chatbota, odpowiedź chatbota na tą wiadomość oraz feedback szefa firmy apropo odpowiedzi chatbota. Twoim zadaniem jest zaktualizować listę obecnych instrukcji aby otrzymany feedback już nieobowiązywał. (szef frimy cofa swój feedback) Dokładnie przeanlizuj konwersację między chatbotem, a klientem oraz feedback szefa i wywnioskuj która reguła z "learened instructions" dotyczy otrzymanego feedbacku i ją usuń.

            ### KONTEKST
            1. **learned instructions:** 
            {row["learned_instructions"]}


            ### LOGIKA AKTUALIZACJI (BARDZO WAŻNE)
            1. Pamiętaj, żeby nie usuwać reguł na siłe, jeśli żadna istniejąca reguła nie pasuje ci do feedbacku to nic nie usuwaj.


            ### ZASADY HIGIENY (MAINTENANCE)
            1. **Konsolidacja:** Jeśli widzisz podobne zasady mówiące o tym samym to połącz je w jedną, zwięzłą zasadę.
            2. **Higienia** Kiedy lista instrukcji robi się długa, to możesz skracać istniejące reguły, ale MUSISZ robić to tak, aby zachowały swój pierwotny sens. Nie rób tego na siłe, tylko kiedy widzisz że jakaś reguły można w oczywisty sposób skrócić. 

            ### Zasady Outputu

            - Oddaj zaktaulizwoną listę 'learned_instructions' 
            - Nie dodawaj żadnych wstępów typu "Oto zaktualizowana lista".
            - Nie używaj formatowania.
            - Ma to być czysty tekst gotowy do zapisania w bazie danych i wrzucony do system prompa chatbota.
            '''
        else:
            service_prompt = f'''
            ### ROLA
            Jesteś Architektem Zachowań AI (AI Behavior Architect). Jesteś absolutnie genialnym prompt engineerem Twoim zadaniem jest utrzymywanie i optymalizacja listy "Wyuczonych Instrukcji" (learned_instructions) dla chatbota biznesowego. Nie jesteś chatbotem – jesteś inżynierem, który go programuje.

            ### CEL
            Otrzymasz informację zwrotną (Feedback) od szefa dotyczącą konkretnej odpowiedzi bota. Otrzymasz wiadomość klienta do chatbota, odpowiedź chatbota na tą wiadomość oraz feedback szefa firmy apropo tej wiadomości. Twoim zadaniem jest zaktualizować listę obecnych instrukcji aby wyeliminować zgłoszony problem. Staraj się myśleć zapobiegawczo, tak żeby instrukcje nie dotyczyły tylko jednej konkretnej sytuacji, ale definiowały ogólne zachowania chatbota w danym typie sytuacji (chyba, że szef w feedbacku jasno zaznaczy inaczej). Dokładnie przeanlizuj konwersację między chatbotem, a klientem oraz  feedback szefa i wywnioskuj co miał dokładnie na myśli, aby zaktualizowane instrukcję rozwiązały problem, który wskazał.

            Otrzymasz również poziom ważności (1-3) 

            ### KONTEKST
            1. obecne **learened instructions:** 
            {row["learned_instructions"]}

            4. **Hard Rules:** {str(row["hard_rules"])} (Tych nie możesz edytować, służą tylko jako kontekst).

            ### LOGIKA AKTUALIZACJI (BARDZO WAŻNE)
            Musisz zastosować odpowiednią strategię w zależności od poziomu ważności:

            - **POZIOM 1 (Drobna sugestia):** Dodaj nową instrukcję lub zedytuj istniejącą jako doprecyzowanie. Nie usuwaj istniejących zasad, chyba że są w 100% sprzeczne.
            - **POZIOM 2 (Istotna zmiana):** Jeśli nowa instrukcja jest sprzeczna z jakąkolwiek starą zasadą, usuń starą i zastąp ją nową. Priorytet ma nowa zasada.
            - **POZIOM 3 (Krytyczna zmiana):** To jest "Hotfix". Nowa instrukcja jest nadrzędna. Usuń wszelkie zasady, które mogłyby jej przeszkadzać. Zmień ton lub styl radykalnie, jeśli tego wymaga feedback.

            Pamiętaj żeby nie dodawać reguł na siłę, jeśli nie jesteś pewny o co chodzi szefowi, to lepiej dodać mniej niż więcej.

            ### ZASADY HIGIENY (MAINTENANCE)
            1. **Konsolidacja:** Jeśli widzisz podobne zasady mówiące o tym samym to połącz je w jedną, zwięzłą zasadę.
            2. **Limit:** Niech tworzone zasady będa krótkie, ale konkretne. Usuwaj zasady trywialne lub zbędne. 
            3. **Przykłady** Nie dodawaj przykładów do stworzonych zasad
            3. **Spójność:** Nowe instrukcje nie mogą przeczyć "Hard Rules" (Sztywnym Regułom).
            4. **Format:** Używaj trybu rozkazującego (np. "Nie używaj emotikon", "Zawsze pytaj o budżet").
            5. **Higienia** Kiedy lista instrukcji robi się długa, to możesz skracać istniejące reguły, ale MUSISZ robić to tak, aby zachowały swój pierwotny sens. Nie rób tego na siłe, tylko kiedy widzisz że jakaś reguły można w oczywisty sposób skrócić. 

            ### Zasady Outputu

            - Oddaj zaktaulizwoną listę 'learned_instructions'
            - Pamiętaj, żeby uwzględnić poprzednie zasady (chyba, że świadomie je usuwasz,)
            - Nie dodawaj żadnych wstępów typu "Oto zaktualizowana lista".
            - Nie używaj formatowania.
            - Ma to być czysty tekst gotowy do zapisania w bazie danych i wrzucony do system prompa chatbota.
        '''
        
        async with OpenRouter(api_key=OPEN_ROUTER) as client:
                response = await client.chat.send_async(
                    model="anthropic/claude-sonnet-4.6",
                    messages=[
                        {
                            "role": "system",
                            "content": service_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_input,
                        }
                    ],
                    temperature=0.2,
                )
    except Exception as e:
        async with OpenRouter(api_key=OPEN_ROUTER) as client:
                response = await client.chat.send_async(
                    model="google/gemini-3.1-pro-preview",
                    messages=[
                        {
                            "role": "system",
                            "content": service_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_input,
                        }
                    ],
                    temperature=0.2,
                )
    
    try:
        await connection.execute('''
        UPDATE ai_persona_config
        SET learned_instructions = $1                             
        WHERE company_name = $2
        ''', response.choices[0].message.content, data.companyName)

        if data.action == "delete":
            await connection.execute('''
        DELETE
        FROM ai_feedback                                                                  
        WHERE id = $1
        ''', data.feedbackId)
        else:
            await connection.execute('''
        UPDATE ai_feedback
        SET status = 'applied'
        WHERE id = $1                                                                
        ''', data.feedbackId)
            
    finally:
        await connection.execute('''
        UPDATE ai_persona_config
        SET is_locked = false
        WHERE company_name = $1
        ''', data.companyName)
        
    return {"status": 200}

    

    




    
  

