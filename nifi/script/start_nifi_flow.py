import os
import time
import requests


NIFI_BASE_URL = os.getenv("NIFI_BASE_URL", "https://localhost:8443/nifi-api")

NIFI_USERNAME = os.getenv("NIFI_USERNAME", "admin")
NIFI_PASSWORD = os.getenv("NIFI_PASSWORD", "adminpassword123")
PROCESS_GROUP_ID = os.getenv("NIFI_PROCESS_GROUP_ID", "root")

TIMEOUT_SECONDS = int(os.getenv("NIFI_TIMEOUT_SECONDS", "180"))


def get_token(session):
    if not NIFI_USERNAME or not NIFI_PASSWORD:
        raise RuntimeError(
            "NIFI_USERNAME e NIFI_PASSWORD devono essere impostati nelle variabili d'ambiente."
        )

    print("Login su NiFi...")

    response = session.post(
        f"{NIFI_BASE_URL}/access/token",
        data={
            "username": NIFI_USERNAME,
            "password": NIFI_PASSWORD,
        },
        headers={
            "Content-Type": "application/x-www-form-urlencoded"
        },
        timeout=15,
        verify=False,
    )

    response.raise_for_status()

    token = response.text
    session.headers.update({"Authorization": f"Bearer {token}"})

    print("Login completato.")


def wait_for_nifi_port(session):
    print("Attendo che NiFi esponga le API...")

    start_time = time.time()

    while time.time() - start_time < TIMEOUT_SECONDS:
        try:
            response = session.get(
                f"{NIFI_BASE_URL}/access/config",
                timeout=10,
                verify=False,
            )

            if response.status_code in (200, 401, 403):
                print("NiFi API raggiungibili.")
                return

        except requests.RequestException:
            pass

        time.sleep(5)

    raise TimeoutError("NiFi non è raggiungibile entro il timeout.")


def wait_for_nifi_authenticated(session):
    print("Verifico accesso autenticato a NiFi...")

    start_time = time.time()

    while time.time() - start_time < TIMEOUT_SECONDS:
        try:
            response = session.get(
                f"{NIFI_BASE_URL}/flow/about",
                timeout=10,
                verify=False,
            )

            if response.status_code == 200:
                about = response.json().get("about", {})
                print(f"NiFi pronto. Versione: {about.get('version')}")
                return

            print(f"NiFi non ancora pronto. HTTP {response.status_code}")

        except requests.RequestException:
            pass

        time.sleep(5)

    raise TimeoutError("NiFi non è diventato disponibile dopo il login.")

def enable_controller_services(session, process_group_id):
    print("Abilito eventuali Controller Services...")

    response = session.put(
        f"{NIFI_BASE_URL}/flow/process-groups/{process_group_id}/controller-services",
        json={
            "id": process_group_id,
            "state": "ENABLED",
        },
        timeout=30,
        verify=False,
    )

    if response.status_code in (200, 202):
        print("Controller Services abilitati.")
    else:
        print("Controller Services non abilitati automaticamente.")
        print(f"HTTP {response.status_code}")
        print(response.text)


def start_process_group(session, process_group_id):
    print(f"Avvio process group: {process_group_id}")

    response = session.put(
        f"{NIFI_BASE_URL}/flow/process-groups/{process_group_id}",
        json={
            "id": process_group_id,
            "state": "RUNNING",
        },
        timeout=30,
        verify=False,
    )

    response.raise_for_status()

    print("Process group avviato correttamente.")


def main():
    session = requests.Session()

    wait_for_nifi_port(session)
    get_token(session)
    wait_for_nifi_authenticated(session)

    enable_controller_services(session, PROCESS_GROUP_ID)
    start_process_group(session, PROCESS_GROUP_ID)

    print("Flow NiFi avviato con successo.")

if __name__ == "__main__":
    main()