import os
import time
import uuid
import requests
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

NIFI_BASE_URL = os.getenv("NIFI_BASE_URL", "https://localhost:8443/nifi-api")

NIFI_USERNAME = os.getenv("NIFI_USERNAME", "admin")
NIFI_PASSWORD = os.getenv("NIFI_PASSWORD", "adminpassword123")

TIMEOUT_SECONDS = int(os.getenv("NIFI_TIMEOUT_SECONDS", "180"))

FLOW_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "..", "flows", "NiFi_Flow.json"
)


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


def get_child_process_group_id(session):
    """Restituisce l'ID del primo process group figlio di root, o None."""
    response = session.get(
        f"{NIFI_BASE_URL}/process-groups/root/process-groups",
        timeout=15,
        verify=False,
    )
    if response.status_code == 200:
        groups = response.json().get("processGroups", [])
        if groups:
            return groups[0]["id"]
    return None


def import_flow(session, flow_json_path):
    print("Importo il flow su NiFi...")

    client_id = str(uuid.uuid4())

    with open(flow_json_path, "rb") as f:
        response = session.post(
            f"{NIFI_BASE_URL}/process-groups/root/process-groups/upload",
            files={"file": ("NiFi_Flow.json", f, "application/json")},
            data={
                "groupName": "NiFi Flow",
                "positionX": 100.0,
                "positionY": 100.0,
                "clientId": client_id,
            },
            timeout=30,
            verify=False,
        )

    if response.status_code in (200, 201):
        pg_id = response.json().get("id")
        print(f"Flow importato correttamente. ID: {pg_id}")
        return pg_id
    else:
        print(f"Errore durante l'import del flow: HTTP {response.status_code}")
        print(response.text)
        raise RuntimeError("Import flow fallito.")


def enable_controller_services(session, process_group_id):
    print(f"Abilito Controller Services nel process group {process_group_id}...")

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
        print(f"Controller Services non abilitati. HTTP {response.status_code}")
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

    pg_id = get_child_process_group_id(session)

    if pg_id:
        print(f"Flow già presente su NiFi (ID: {pg_id}), skip import.")
    else:
        pg_id = import_flow(session, FLOW_JSON_PATH)
        # Attendi che NiFi registri i controller services
        time.sleep(3)

    enable_controller_services(session, pg_id)

    # Attendi che i services siano effettivamente abilitati
    time.sleep(3)

    # Avvia il process group figlio
    start_process_group(session, pg_id)

    print("Flow NiFi avviato con successo.")

if __name__ == "__main__":
    main()
